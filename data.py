import os

import lightning as L
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split


class SingleCellDataset(Dataset):
    def __init__(self, data_path: str, label_rep: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        adata = sc.read_h5ad(data_path)
        self.X = torch.from_numpy(adata.X).float()  # .to(self.device)
        self.label_encoder = LabelEncoder()
        self.y = torch.from_numpy(self.label_encoder.fit_transform(adata.obs[label_rep].values))  # .to(self.device)
        del adata

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"expr": self.X[idx,], "label": self.y[idx]}


class SingleCellDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        label_rep: str,
        batch_size: int = 256,
        train_size: float = 0.7,
        val_size: float = 0.2,
        num_workers: int = os.cpu_count() // 2,
    ):
        super().__init__()
        assert (train_size + val_size) <= 1, ValueError("sum of sizes must be <= 1.")
        self.train_size, self.val_size = train_size, val_size
        self.test_size = 1 - train_size - val_size
        self.data_path = data_path
        self.label_rep = label_rep
        self.batch_size = batch_size
        self.num_workers = num_workers

        scdataset = SingleCellDataset(self.data_path, self.label_rep)
        train_ds, val_ds, test_ds = random_split(scdataset, [self.train_size, self.val_size, self.test_size])
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        self.label_encoder = scdataset.label_encoder

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(  # set batch size == 1 to enable the skipping inference
            self.test_dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True
        )


# train_loader, val_loader, test_loader = load_data("datasets/covid19_GSE158055_preprocessed_health.h5ad", "majorType")

# for i, samples in enumerate(train_loader):
#     print(i, samples["expr"].shape, samples["label"].device)

# print("done!")
