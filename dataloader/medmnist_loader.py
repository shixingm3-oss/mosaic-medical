"""
MedMNIST data loading utilities.

Wraps the official ``medmnist`` package and adds dual-augmentation support
for teacher-student training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

import medmnist
from medmnist import INFO

from config.datasets import get_dataset_config
from .transforms import DualTransform2D, DualTransform3D, get_transforms


class MedMNISTDataset(Dataset):
    """Thin wrapper around the official MedMNIST datasets."""

    def __init__(self, dataset_name, split="train", data_root="./data",
                 dual_transform=False, transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.dual_transform = dual_transform
        self.config = get_dataset_config(dataset_name)
        self.is_3d = self.config.is_3d

        info = INFO[self.config.medmnist_name]
        DataClass = getattr(medmnist, info["python_class"])
        kw = dict(split=split, transform=None, download=False, root=data_root)
        if self.is_3d:
            self.dataset = DataClass(size=64, **kw)
        else:
            self.dataset = DataClass(size=224, as_rgb=True, **kw)

        if transform is not None:
            self.transform = transform
        elif dual_transform and split == "train":
            self.transform = DualTransform3D() if self.is_3d else DualTransform2D()
        else:
            self.transform = get_transforms(self.is_3d, split, dual=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label = torch.tensor(label).squeeze()
        if self.dual_transform and self.split == "train":
            return self.transform(img), label
        return self.transform(img), label


def create_dataloader(dataset_name, split="train", data_root="./data",
                      batch_size=32, num_workers=4, dual_transform=False,
                      shuffle=None, pin_memory=True):
    ds = MedMNISTDataset(dataset_name, split, data_root, dual_transform)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory,
                      drop_last=(split == "train"))


def create_all_dataloaders(dataset_list: List[str], data_root="./data",
                           batch_size=32, num_workers=4,
                           dual_transform=False) -> Dict:
    loaders = {}
    for name in dataset_list:
        loaders[name] = {
            "train": create_dataloader(name, "train", data_root, batch_size,
                                       num_workers, dual_transform),
            "val":   create_dataloader(name, "val", data_root, batch_size,
                                       num_workers, False),
            "test":  create_dataloader(name, "test", data_root, batch_size,
                                       num_workers, False),
        }
        ds = loaders[name]
        print(f"  {name}: train={len(ds['train'].dataset)}  "
              f"val={len(ds['val'].dataset)}  test={len(ds['test'].dataset)}")
    return loaders