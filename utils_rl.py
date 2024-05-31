import os, argparse, warnings
import torch
import torch.nn.functional as F
import lightning as L
import transformers
import ray
from typing import List, Literal
from collections import deque
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1_score, confusion_matrix
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from tqdm import trange, tqdm

from data import SingleCellDataModule
from models import PretrainedGatedTransformerEncoderClassifier


class BatchCrossEntropy(nn.Module):
    """Per sample loss."""

    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        # x shape: (batch_size, num_celltypes)
        logp = F.log_softmax(x, dim=1)
        target = target.view(-1, 1)
        output = -logp.gather(1, target)
        return output


class Trainer:
    def __init__(
        self,
        max_epochs: int,
        alpha: float,
        beta: float,
        gamma: float,  # discount factor
        lr: float,
        wd: float,
        lr_scheduler_type: Literal["constant", "linear", "cosine", "polynomial"] = "constant",
        warmup_steps: int = 0,
        default_root_dir: str = "ckpt/reinforce/",
    ) -> None:
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.wd = wd
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps

        self.criterion = BatchCrossEntropy().cuda()
        self.total_criterion = nn.CrossEntropyLoss().cuda()

        self.gate_rewards = []  # store detached tensors
        self.saved_log_probs = []
        self.gate_saved_actions = []

        previous_versions = [
            int(f.split("_")[1])
            for f in os.listdir(default_root_dir)
            if f.startswith("version_") and os.path.isdir(os.path.join(default_root_dir, f))
        ]
        new_version = "version_0" if len(previous_versions) == 0 else f"version_{max(previous_versions) + 1}"
        new_version_dir = os.path.join(default_root_dir, new_version)
        os.makedirs(new_version_dir)
        self.writer = SummaryWriter(log_dir=new_version_dir)

        self.best_score = 0
        self.best_model_dir = None

        self.save_hyperparameters()

    def save_hyperparameters(self):
        self.writer.add_hparams(
            hparam_dict={
                "max_epochs": self.max_epochs,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "lr": self.lr,
                "wd": self.wd,
                "lr_scheduler_type": self.lr_scheduler_type,
                "warmup_steps": self.warmup_steps,
            },
            metric_dict={"best_f1": self.best_score},
            run_name=".",  # to prevent create a new run
        )

    def select_action(self, probs: List[Tensor]):
        """Sve the log probabilities and actions of each gate"""
        for prob in probs:
            m = torch.distributions.Bernoulli(prob)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            self.gate_saved_actions.append(action)

    def finish_episode(self, init_R: Tensor, ce_loss: Tensor):
        R = init_R
        policy_loss = []
        returns = deque()
        for r in self.gate_rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        # returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        totoal_loss = policy_loss + ce_loss  # ce_loss  # policy_loss + ce_loss
        totoal_loss.backward()  # optimize hybrid loss
        self.optimizer.step()
        self.scheduler.step()

        cur_step = int(self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"])
        self.writer.add_scalar("loss/policy", policy_loss.detach(), cur_step)
        self.writer.add_scalar("loss/supervised", ce_loss.detach(), cur_step)
        self.writer.add_scalar("loss/hybrid", totoal_loss.detach(), cur_step)
        self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], cur_step)
        self.gate_rewards.clear()
        self.saved_log_probs.clear()

    def configure_optimizer(self, model: L.LightningModule, training_steps: int):
        self.optimizer = optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()), lr=self.lr, weight_decay=self.wd
        )
        if self.lr_scheduler_type == "constant":
            self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, self.warmup_steps)
        elif self.lr_scheduler_type == "linear":
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, self.warmup_steps, training_steps
            )
        elif self.lr_scheduler_type == "cosine":
            self.scheduler = transformers.get_cosine_schedule_with_warmup(
                self.optimizer, self.warmup_steps, training_steps
            )
        elif self.lr_scheduler_type == "polynomial":
            self.scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                self.optimizer, self.warmup_steps, training_steps
            )
        else:
            raise ValueError(
                "The argument `lr_scheduler_type` can only be one of 'constant', 'linear', 'cosine', or 'polynomial'."
            )

    def train(self, datamodule: L.LightningDataModule, model: L.LightningModule):
        warnings.filterwarnings("ignore", category=UserWarning)

        # setup model
        model.on_train_epoch_start()
        model.train()
        num_encoders = len(model.model.encoder)

        # setup dataloaders
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        num_batches = len(train_dataloader)
        num_training_steps = self.max_epochs * num_batches

        # setup optimizer and schedulerc
        self.configure_optimizer(model, num_training_steps)

        # normalize hyperparameters `alpha` and `beta`
        normalized_alpha = self.alpha / num_encoders
        normalized_beta = self.beta / num_encoders

        for step in trange(num_training_steps):
            batch_data = next(iter(train_dataloader))
            x, y = batch_data["expr"].cuda(), batch_data["label"].cuda()
            y_hat, masks, probs = model(x)

            # skips = [mask.detach().le(0.5).float().mean() for mask in masks]  # skip ratios of every gate
            pred_loss = self.criterion(y_hat, y)
            ce_loss = self.total_criterion(y_hat, y)
            self.select_action(probs)

            # get rewards
            for act in self.gate_saved_actions:
                self.gate_rewards.append((1 - act.float()).detach() * normalized_alpha)
            self.finish_episode(-pred_loss.detach() * normalized_beta, ce_loss)

            if (step + 1) % num_batches == 0:  # the end of a epoch
                cur_epoch = step // num_batches
                metrics = self.evaluate(val_dataloader, model, leave=False)
                self.writer.add_scalar("val/acc", metrics["acc"], cur_epoch)
                self.writer.add_scalar("val/f1", metrics["f1"], cur_epoch)
                self.save_best_model(metrics["f1"], cur_epoch, model)

        self.writer.close()

    def save_best_model(self, score: float, current_epoch: int, model: L.LightningModule):
        if score > self.best_score:
            self.best_score = score
            best_model_name = f"epoch={current_epoch}_best_score={score:.4f}.pth"
            if self.best_model_dir is not None:
                os.remove(self.best_model_dir)
            self.best_model_dir = os.path.join(self.writer.log_dir, best_model_name)
            torch.save(model.state_dict(), self.best_model_dir)

    def evaluate(
        self,
        dataloader: DataLoader,
        model: L.LightningModule,
        stream: bool = False,
        verbose: bool = False,
        leave: bool = True,
    ):
        if not stream:
            model.on_validation_start()
        model.eval()
        y_trues, y_hats = [], []
        # masks_sum = []

        for sample in tqdm(dataloader, leave=leave):
            x, y = sample["expr"].cuda(), sample["label"].cuda()
            y_hat, masks, probs = model(x)
            y_hats.append(y_hat.detach())
            y_trues.append(y.detach())
            # masks_sum.append(torch.sum(torch.cat(masks)).detach())

        y_hat = torch.cat(y_hats)
        y_true = torch.cat(y_trues)
        # num_executed_blocks = torch.sum(torch.tensor(masks_sum))

        acc = accuracy(y_hat, y_true, task="multiclass", num_classes=model.num_celltypes).item()
        f1 = f1_score(y_hat, y_true, task="multiclass", num_classes=model.num_celltypes, average="macro").item()
        conf_mtx = confusion_matrix(y_hat, y_true, task="multiclass", num_classes=model.num_celltypes)
        if verbose:
            print(f"Accuracy score: {acc} | Macro F1 score: {f1}")
        return {"acc": acc, "f1": f1, "conf_mtx": conf_mtx}


def single_run(config: dict):

    dm = SingleCellDataModule(config["dataset_dir"], config["celltype_rep"])
    num_classes = dm.label_encoder.classes_.shape[0]
    model = PretrainedGatedTransformerEncoderClassifier(num_celltypes=num_classes).cuda()

    trainer = Trainer(
        max_epochs=config["epochs"],
        alpha=config["alpha"],
        beta=config["beta"],
        gamma=config["gamma"],
        lr=config["lr"],
        wd=config["wd"],
        lr_scheduler_type=config["scheduler"],
        warmup_steps=config["warmup_steps"],
        default_root_dir=config["root_dir"],
    )
    if config["saved_dir"] is None:  # train the model from scratch and test its performance
        pretrained_ckpt = torch.load(config["pretrained_dir"])
        model.load_state_dict(pretrained_ckpt["state_dict"])
        trainer.train(dm, model=model)
        if config["n_trials"]:
            metrics = trainer.evaluate(dataloader=dm.test_dataloader(), model=model, verbose=False)
            train.report({"macro_f1": trainer.best_score})
        else:
            trainer.evaluate(dataloader=dm.test_dataloader(), model=model, verbose=True)
    else:
        model.load_state_dict(torch.load(config["saved_dir"]))
        if config["stream"]:
            trainer.evaluate(dataloader=dm.predict_dataloader(), model=model, verbose=True, stream=True)
        else:
            trainer.evaluate(dataloader=dm.test_dataloader(), model=model, verbose=True)


def hparam_search(args: argparse.Namespace):
    ray.init()
    ray_config = vars(args)
    ray_config["alpha"] = tune.loguniform(1e-4, 1e-1)
    ray_config["beta"] = tune.choice([1e-2, 1e-1, 1])
    ray_config["gamma"] = tune.choice([0.5, 0.7, 0.9, 0.99])
    ray_config["lr"] = tune.loguniform(1e-5, 1e-2)
    ray_config["wd"] = tune.uniform(0.0, 0.3)
    ray_config["scheduler"] = tune.choice(["constant", "linear", "cosine", "polynomial"])
    ray_config["warmup_steps"] = tune.randint(100, 2000)

    ray_config["pretrained_dir"] = os.path.abspath(ray_config["pretrained_dir"])
    ray_config["dataset_dir"] = os.path.abspath(ray_config["dataset_dir"])
    ray_config["root_dir"] = os.path.abspath(ray_config["root_dir"])

    algo = OptunaSearch()
    single_run_with_resource = tune.with_resources(single_run, {"gpu": 1})
    tuner = tune.Tuner(
        single_run_with_resource,
        tune_config=tune.TuneConfig(metric="macro_f1", mode="max", search_alg=algo, num_samples=args.n_trials),
        param_space=ray_config,
        run_config=train.RunConfig(storage_path=ray_config["root_dir"], name="optuna"),
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
