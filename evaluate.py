"""Compute PSNR / SSIM on the validation set and save a comparison image.
python evaluate.py --data_dir data/flowers"""
import argparse

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset import ColorDataset, get_splits
from utils import load_model, tensors_to_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/flowers")
    p.add_argument("--weights", default="model/colorizer.pth")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    _, val_files = get_splits(args.data_dir)
    dl = DataLoader(ColorDataset(val_files, train=False), batch_size=32)

    scores = {"model": ([], []), "grayscale baseline": ([], [])}
    grid_rows = []
    with torch.no_grad():
        for L, ab in dl:
            pred = model(L.to(device)).cpu()
            real = tensors_to_rgb(L, ab)
            fake = tensors_to_rgb(L, pred)
            gray = tensors_to_rgb(L, torch.zeros_like(ab))
            for r, f, g in zip(real, fake, gray):
                for name, img in (("model", f), ("grayscale baseline", g)):
                    scores[name][0].append(psnr(r, img, data_range=255))
                    scores[name][1].append(ssim(r, img, channel_axis=-1, data_range=255))
                if len(grid_rows) < 6:
                    grid_rows.append(np.concatenate([g, f, r], axis=1))

    lines = [f"Validation images: {len(val_files)}", ""]
    for name, (ps, ss) in scores.items():
        lines.append(f"{name:20s} PSNR {np.mean(ps):6.2f} dB | SSIM {np.mean(ss):.4f}")
    text = "\n".join(lines)
    print(text)
    open("results/metrics.txt", "w").write(text + "\n")
    Image.fromarray(np.concatenate(grid_rows, axis=0)).save("results/comparison.png")
    print("Saved results/metrics.txt and results/comparison.png (columns: gray | model | original)")


if __name__ == "__main__":
    main()
