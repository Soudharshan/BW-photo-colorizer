"""Dataset that turns any folder of colour photos into (L, ab) training pairs."""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from utils import IMG_SIZE, lab_to_tensors, pil_to_lab

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_splits(root, val_frac=0.1, seed=42):
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in '{root}'. Run download_data.py or add photos.")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    return files[n_val:], files[:n_val]


class ColorDataset(Dataset):
    def __init__(self, files, size=IMG_SIZE, train=True):
        self.files, self.size, self.train = files, size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        if self.train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return lab_to_tensors(pil_to_lab(img))
