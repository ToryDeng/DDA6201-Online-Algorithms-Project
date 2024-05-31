"""Supevised pre-training"""

import os, argparse, warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# filter the UserWarning: Plan failed with a cudnnException in pytorch 2.3
warnings.filterwarnings("ignore", category=UserWarning)

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

from data import SingleCellDataModule
from models import PretrainedGatedTransformerEncoderClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the random seed (default: 42)", default=42)
parser.add_argument("--smoke_test", help="Whether to perform a smoke test", action="store_true")
parser.add_argument(
    "--saved_dir",
    help="Path to the trained model, which will be used to predict samples in the test dataset",
    required=False,
)
parser.add_argument("--stream", help="Whether to predict a single sample each batch", action="store_true")
args = parser.parse_args()
if args.smoke_test:
    epochs, limit_train_batches, limit_val_batches, log_every_n_steps = 2, 20, 10, 2
else:
    epochs, limit_train_batches, limit_val_batches, log_every_n_steps = 50, 1.0, 1.0, 50

L.seed_everything(args.seed, workers=True)
torch.set_float32_matmul_precision("high")
model_checkpoint = ModelCheckpoint(
    monitor="val_f1", mode="max", verbose=False, filename="{epoch}_{val_f1:.4f}", save_last=True, save_top_k=3
)
model_summary = RichModelSummary(max_depth=2)
progress_bar = RichProgressBar()
trainer = L.Trainer(
    devices=1,
    max_epochs=epochs,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    log_every_n_steps=log_every_n_steps,
    callbacks=[model_checkpoint, model_summary, progress_bar],
    deterministic=True,
    fast_dev_run=False,
    default_root_dir="ckpt/pretrained/",
)
dm = SingleCellDataModule("datasets/covid19_GSE158055_preprocessed_health.h5ad", "majorType")
num_classes = dm.label_encoder.classes_.shape[0]
model = PretrainedGatedTransformerEncoderClassifier(num_celltypes=num_classes, label_encoder=dm.label_encoder)
if args.saved_dir is None:
    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
else:
    if args.stream:
        trainer.test(model=model, ckpt_path=args.saved_dir, dataloaders=dm.predict_dataloader(), verbose=False)
    else:
        trainer.test(model=model, ckpt_path=args.saved_dir, dataloaders=dm.test_dataloader(), verbose=True)
