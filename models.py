import math

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from sklearn.preprocessing import LabelEncoder
from torch import Tensor, nn, optim
from torchmetrics.functional import accuracy, f1_score, confusion_matrix

plt.rcParams["font.family"] = "Times New Roman"


class ContinuousValueEncoder(nn.Module):
    """
    Customized embedding module.
    Encode real number values to a vector using neural nets projection.
    Copied from https://github.com/bowang-lab/scGPT/blob/main/scgpt/model/model.py#L765.
    The default value of `max_value` is changed from 512 to 10.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape (batch_size, seq_len)
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class VanillaTransformerEncoderClassifier(L.LightningModule):
    """
    Vanilla base classifier.
    """

    def __init__(
        self,
        num_celltypes: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        fc_dim: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        label_encoder: LabelEncoder = None,
    ):
        super().__init__()
        self.num_celltypes = num_celltypes
        self.learning_rate = learning_rate
        self.emb = ContinuousValueEncoder(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=fc_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, num_celltypes)
        self.dropout = nn.Dropout(p=dropout)

        self.y_hats = []
        self.y_trues = []
        self.label_encoder = label_encoder

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)  # (batch_size, num_genes, embed_dim)
        x = self.encoder(x)
        x = self.dropout(x)
        x = x.max(dim=1)[0]  # or mean
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # torchmetrics will automatically convert from logits to indices
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_celltypes)
        f1 = f1_score(y_hat, y, task="multiclass", num_classes=self.num_celltypes, average="macro")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat = self(x)

        self.y_hats.append(y_hat)
        self.y_trues.append(y)

    def on_test_epoch_end(self):
        y_hat = torch.cat(self.y_hats)
        y_true = torch.cat(self.y_trues)
        # accuracy
        acc = accuracy(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes)
        self.log("test_acc", acc, logger=True)
        # macro F1 score
        f1 = f1_score(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes, average="macro")
        self.log("test_f1", f1, logger=True)
        # confusion matrix
        conf_mtx = confusion_matrix(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes)
        fig = plt.figure(figsize=(8, 7), dpi=400)
        celltypes = self.label_encoder.classes_
        kwargs = {"annot": True, "square": True, "fmt": "g", "xticklabels": celltypes, "yticklabels": celltypes}
        ax = sns.heatmap(conf_mtx.cpu(), cmap="viridis", linewidths=1, **kwargs)
        ax.set(xlabel="Prediction", ylabel="Ground truth")
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        # clean memory
        self.y_hats.clear()
        self.y_trues.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class GatedTransformerEncoderClassifier(nn.Module):
    """
    Pure pytorch model.
    """

    def __init__(
        self,
        num_celltypes: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        fc_dim: int = 128,
        pool_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.emb = ContinuousValueEncoder(embed_dim)
        self.encoder = nn.ModuleList(
            [
                GatedTransformerEncoderLayer(
                    embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, pool_size=pool_size
                ).cuda()  # TODO: check why deepcopy + MouduleList will cause ascending trend in val loss
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(embed_dim, num_celltypes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)  # (batch_size, num_genes, embed_dim)
        masks, probs = [], []
        for module in self.encoder:
            x, mask, prob = module(x)  # `probs` contains the outputs from the gate after sigmoid
            masks.append(mask)
            probs.append(prob)
        x = self.dropout(x)
        x = x.max(dim=1)[0]  # or mean
        out = self.linear(x)
        return out, masks, probs


class PretrainedGatedTransformerEncoderClassifier(L.LightningModule):
    """
    Step 1: Supervised Pretraining model.
    """

    def __init__(
        self,
        num_celltypes: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        fc_dim: int = 128,
        pool_size: int = 5,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        label_encoder: LabelEncoder = None,
    ):
        super().__init__()
        self.num_celltypes = num_celltypes
        self.learning_rate = learning_rate
        self.model = GatedTransformerEncoderClassifier(
            num_celltypes, embed_dim, num_layers, num_heads, fc_dim, pool_size, dropout
        )

        self.y_hats = []
        self.y_trues = []
        self.label_encoder = label_encoder

    def on_train_epoch_start(self) -> None:
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = True

    def on_train_epoch_end(self) -> None:
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = False

    def on_validation_start(self) -> None:
        """Allow batch computation at validation start"""
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = True

    def on_validation_end(self) -> None:
        """forbid batch computation in validation end"""
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = False

    def on_test_epoch_end(self):
        y_hat = torch.cat(self.y_hats)
        y_true = torch.cat(self.y_trues)
        # accuracy
        acc = accuracy(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes)
        self.log("test_acc", acc, logger=True)
        # macro F1 score
        f1 = f1_score(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes, average="macro")
        self.log("test_f1", f1, logger=True)
        # confusion matrix
        conf_mtx = confusion_matrix(y_hat, y_true, task="multiclass", num_classes=self.num_celltypes)
        fig = plt.figure(figsize=(8, 7), dpi=400)
        celltypes = self.label_encoder.classes_
        kwargs = {"annot": True, "square": True, "fmt": "g", "xticklabels": celltypes, "yticklabels": celltypes}
        ax = sns.heatmap(conf_mtx.cpu(), cmap="viridis", linewidths=1, **kwargs)
        ax.set(xlabel="Prediction", ylabel="Ground truth")
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        # clean memory
        self.y_hats.clear()
        self.y_trues.clear()

    def on_test_start(self) -> None:
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = True

    def on_test_end(self) -> None:
        for encoder_layer in self.model.encoder:
            encoder_layer.allow_batch = False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat, gate_masks, gate_probs = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat, gate_masks, gate_probs = self(x)
        loss = F.cross_entropy(y_hat, y)
        # torchmetrics will automatically convert from logits to indices
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_celltypes)
        f1 = f1_score(y_hat, y, task="multiclass", num_classes=self.num_celltypes, average="macro")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch["expr"], batch["label"]
        y_hat, gate_masks, gate_probs = self(x)

        self.y_hats.append(y_hat)
        self.y_trues.append(y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        for name, layer in self.model.named_modules():
            if "gate" in name:
                norms = grad_norm(layer, norm_type=2)
                self.log_dict(norms)


class ConvGate(nn.Module):
    def __init__(self, pool_size: int, num_channels: int):
        super().__init__()
        self.pool_size = pool_size
        self.num_channels = num_channels
        # require input shape (batch_size, num_channels, seq_len); num_channels == embed_dim
        self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm1d(num_channels)
        self.activation = nn.ReLU()

        pool_size = math.floor(pool_size / 2 + 0.5)  # for conv stride = 2
        self.pool = nn.AvgPool1d(pool_size)

        # self.linear = nn.LazyLinear(out_features=2)
        self.linear = nn.Linear(in_features=num_channels, out_features=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        x : Tensor
            Input tensors with shape (batch_size, seq_len, num_channels).
        """
        x = x.permute(0, 2, 1)  # exchange dim 2 and 1
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.mean(dim=2)  # (batch_size, num_channels)
        x = self.linear(x)  # (batch_size, 1)
        prob = F.sigmoid(x)  # (batch_size, 1)
        # .detach() ensures the different forward / backward passes
        disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob  # (batch_size, 1)
        return disc_prob, prob


class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, fc_dim: int, pool_size: int):
        super().__init__()
        # require input shape (batch_size, seq_len, num_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=fc_dim, batch_first=True
        )
        self.gate = ConvGate(pool_size=pool_size, num_channels=embed_dim)
        self.allow_batch = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if not self.allow_batch:  # the samples must be predicted one by one
            if x.shape[0] != 1:  # batch size != 1
                raise ValueError("Batch size must be 1 when the model is not under training.")
            else:  # batch size == 1
                mask, prob = self.gate(x)
                if mask == 1:  # 1 means executing the current encoder layer
                    out = self.encoder_layer(x)
                    return out, mask, prob
                else:
                    out = x
                    return out, mask, prob  # `out`: 0 means skipping the current encoder layer
        else:  # allow batch computation in training
            mask, prob = self.gate(x)
            out = mask.unsqueeze(2) * self.encoder_layer(x) + (1 - mask.unsqueeze(2)) * x  # approximate
            return out, mask, prob


# torch.manual_seed(2)
# inputs = torch.randn((1, 2000)).to("cuda")
# model = PretrainedGatedTransformerEncoderClassifier(num_celltypes=8).to("cuda")

# model = ConvGate(3, 50).to("cuda")
# for encoder_layer in model.encoder:
#     encoder_layer.allow_batch = True
# y_hat, mask, prob = model(inputs)
# print(y_hat.shape)
# y = torch.randint(5, (32,), dtype=torch.int64).to("cuda")
# print(y.shape, y_hat.shape)
# print(F.cross_entropy(y_hat, y))
# model = TransformerEncoderClassifierWithGate(num_celltypes=10).to("cuda")
# model.eval()
# out = model(inputs)
# print(out.shape)
