"""Train the colourizer.   python train.py --data_dir data/flowers --epochs 30"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from dataset import ColorDataset, get_splits
from model import ColorizerNet
from utils import tensors_to_rgb


def save_samples(model, loader, device, path):
    model.eval()
    L, ab = next(iter(loader))
    L, ab = L[:8].to(device), ab[:8].to(device)
    with torch.no_grad():
        pred = model(L)
    gray = tensors_to_rgb(L, torch.zeros_like(ab))
    fake = tensors_to_rgb(L, pred)
    real = tensors_to_rgb(L, ab)
    rows = [np.concatenate([g, f, r], axis=1) for g, f, r in zip(gray, fake, real)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)  # gray | predicted | original


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/flowers")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_files, val_files = get_splits(args.data_dir)
    print(f"Train images: {len(train_files)} | Val images: {len(val_files)}")
    train_dl = DataLoader(ColorDataset(train_files, train=True), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.workers, drop_last=len(train_files) >= args.batch_size)
    val_dl = DataLoader(ColorDataset(val_files, train=False), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)

    model = ColorizerNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.L1Loss()   # L1 gives sharper colours than MSE (MSE averages to grey/brown)

    Path("model").mkdir(exist_ok=True)
    Path("samples").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    history, best = [], float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr, n = 0.0, 0
        for L, ab in train_dl:
            L, ab = L.to(device), ab.to(device)
            opt.zero_grad()
            loss = loss_fn(model(L), ab)
            loss.backward()
            opt.step()
            tr += loss.item() * L.size(0)
            n += L.size(0)
        tr /= n

        model.eval()
        va = 0.0
        with torch.no_grad():
            for L, ab in val_dl:
                L, ab = L.to(device), ab.to(device)
                va += loss_fn(model(L), ab).item() * L.size(0)
        va /= len(val_dl.dataset)
        sched.step()

        history.append((epoch, tr, va))
        print(f"Epoch {epoch:3d}/{args.epochs} | train L1 {tr:.4f} | val L1 {va:.4f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), "model/colorizer.pth")
        if epoch % 5 == 0 or epoch == 1:
            save_samples(model, val_dl, device, f"samples/epoch_{epoch:03d}.png")

    with open("results/history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_l1", "val_l1"])
        w.writerows(history)
    ep, trl, val = zip(*history)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, trl, label="train")
    plt.plot(ep, val, label="validation")
    plt.xlabel("epoch"); plt.ylabel("L1 loss"); plt.legend(); plt.title("Training curve")
    plt.tight_layout(); plt.savefig("results/loss_curve.png", dpi=150)
    print(f"Done. Best val L1 = {best:.4f}. Weights saved to model/colorizer.pth")


if __name__ == "__main__":
    main()
