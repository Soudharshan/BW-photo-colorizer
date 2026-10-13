"""Helpers for Lab colour conversion and colourizing images."""
import numpy as np
import torch
from PIL import Image
from skimage import color

IMG_SIZE = 128        # model input size (must be divisible by 16)
AB_SCALE = 110.0      # a/b channels are divided by this -> roughly [-1, 1]


def pil_to_lab(img, size=None):
    """PIL image -> float32 Lab array (H, W, 3)."""
    img = img.convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    return color.rgb2lab(arr)


def lab_to_tensors(lab):
    """Lab array -> (L, ab) tensors. L in [0,1], ab in about [-1,1]."""
    L = np.ascontiguousarray(lab[..., 0:1] / 100.0).transpose(2, 0, 1)
    ab = np.ascontiguousarray(lab[..., 1:] / AB_SCALE).transpose(2, 0, 1)
    return torch.from_numpy(L).float(), torch.from_numpy(ab).float()


def tensors_to_rgb(L, ab):
    """Batch of L (B,1,H,W) and ab (B,2,H,W) tensors -> uint8 RGB (B,H,W,3)."""
    L = L.detach().cpu().numpy().transpose(0, 2, 3, 1) * 100.0
    ab = ab.detach().cpu().numpy().transpose(0, 2, 3, 1) * AB_SCALE
    lab = np.concatenate([L, ab], axis=-1)
    out = []
    for x in lab:
        rgb = np.clip(color.lab2rgb(x), 0, 1)
        out.append((rgb * 255).astype(np.uint8))
    return np.stack(out)


@torch.no_grad()
def colorize_image(model, img, device="cpu", size=IMG_SIZE):
    """Colourize a PIL image of any size.

    The model predicts colour (ab) at low resolution; we upsample the colour
    and combine it with the ORIGINAL full-resolution L channel, so the
    photo keeps all its sharpness.
    """
    model.eval()
    img = img.convert("RGB")
    w, h = img.size

    L_full = pil_to_lab(img)[..., 0]
    lab_small = pil_to_lab(img, size=size)
    L_in = torch.from_numpy(np.ascontiguousarray(lab_small[..., 0] / 100.0)).float()
    L_in = L_in[None, None].to(device)

    ab_pred = model(L_in)[0].cpu().numpy().transpose(1, 2, 0) * AB_SCALE  # (s,s,2)
    ab_up = np.stack(
        [np.array(Image.fromarray(ab_pred[..., i]).resize((w, h), Image.BICUBIC))
         for i in range(2)],
        axis=-1,
    )
    lab_out = np.concatenate([L_full[..., None], ab_up], axis=-1)
    rgb = np.clip(color.lab2rgb(lab_out), 0, 1)
    return Image.fromarray((rgb * 255).astype("uint8"))


def load_model(path, device="cpu"):
    from model import ColorizerNet
    net = ColorizerNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
