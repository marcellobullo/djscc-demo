"""Dataset utilities for training/evaluating DJSCC models.

Supports flat image directories (like DIV2K) and ImageFolder layouts.
Multiple directories can be combined into a single dataset.
"""

import os
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from PIL import Image

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDirectoryDataset(Dataset):
    """Loads images from a flat directory (no subfolder structure needed).

    Works with DIV2K, Flickr2K, CLIC, or any folder of images.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            p for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def build_combined_dataset(
    dirs: list[str],
    transform=None,
) -> Dataset:
    """Combine images from multiple directories into one dataset.

    Each directory can be either flat (DIV2K-style) or ImageFolder-style.
    """
    datasets = []
    for d in dirs:
        subdirs = [
            e for e in Path(d).iterdir()
            if e.is_dir() and any(
                f.suffix.lower() in IMG_EXTENSIONS for f in e.iterdir() if f.is_file()
            )
        ]
        if subdirs:
            from torchvision.datasets import ImageFolder
            datasets.append(ImageFolder(d, transform))
        else:
            datasets.append(ImageDirectoryDataset(d, transform))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_train_loader(
    data_dirs: list[str],
    img_size: tuple[int, int],
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    w, h = img_size
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    ds = build_combined_dataset(data_dirs, transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def build_val_loader(
    data_dirs: list[str],
    img_size: tuple[int, int],
    batch_size: int = 1,
    num_workers: int = 2,
) -> DataLoader:
    w, h = img_size
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    ds = build_combined_dataset(data_dirs, transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
