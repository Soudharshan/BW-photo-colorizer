"""Downloads the Oxford Flowers102 dataset (~330 MB) and saves it as JPGs in data/flowers/.
You can skip this and put ANY folder of colour photos in data/ instead
(landscapes, animals, etc.). One category gives the best results."""
from pathlib import Path

from torchvision.datasets import Flowers102

out = Path("data/flowers")
out.mkdir(parents=True, exist_ok=True)
n = 0
for split in ["train", "val", "test"]:
    ds = Flowers102(root="data/_raw", split=split, download=True)
    for img, _ in ds:
        img.convert("RGB").save(out / f"{n:05d}.jpg", quality=95)
        n += 1
print(f"Saved {n} images to {out}")
