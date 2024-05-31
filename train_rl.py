"""Reinforcement learning"""

import argparse
import os
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# filter the UserWarning: Plan failed with a cudnnException in pytorch 2.3
warnings.filterwarnings("ignore", category=UserWarning)

import lightning as L
import torch

from utils_rl import hparam_search, single_run

torch.use_deterministic_algorithms(True)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, help="the random seed (default: 42)", default=42)
parser.add_argument("--lr", type=float, help="the learning rate", default=1e-3)
parser.add_argument("--wd", type=float, help="the weight decay", default=0)
parser.add_argument(
    "--scheduler",
    type=str,
    help="the learning rate scheduler type",
    choices=["constant", "linear", "cosine", "polynomial"],
    default="constant",
)
parser.add_argument("--warmup_steps", type=int, help="the warmup steps", default=0)
parser.add_argument("--epochs", type=int, help="the number of epochs", default=50)
parser.add_argument("--alpha", type=float, help="the alpha hyperparameter", default=0.1)  # should be tuned
parser.add_argument("--beta", type=float, help="the beta hyperparameter", default=0.1)  # should be tuned
parser.add_argument("--gamma", type=float, help="the discount factor", default=0.99)
parser.add_argument(
    "--pretrained_dir",
    type=str,
    help="the discount factor",
    default="./ckpt/pretrained/lightning_logs/version_0/checkpoints/epoch=47_val_f1=0.6034.ckpt",
)
parser.add_argument(
    "--saved_dir",
    help="Path to the trained model, which will be used to predict samples in the test dataset",
    required=False,
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Path to the dataset",
    default="./datasets/covid19_GSE158055_preprocessed_health.h5ad",
)
parser.add_argument("--root_dir", type=str, help="Default root directory", default="./ckpt/reinforce/")
parser.add_argument(
    "--celltype_rep",
    type=str,
    help="Name of the column containing cell type labels in the AnnData.obs",
    default="majorType",
)
parser.add_argument("--stream", help="Whether to predict a single sample each batch", action="store_true")
parser.add_argument("--n_trials", type=int, help="the number of trials used in hyperparameter optimization", default=0)
args = parser.parse_args()


L.seed_everything(args.seed)

if args.n_trials:
    hparam_search(args)
else:
    single_run(vars(args))
