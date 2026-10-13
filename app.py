"""Streamlit demo.   streamlit run app.py"""
import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from utils import colorize_image, load_model

WEIGHTS = "model/colorizer.pth"
MAX_SIDE = 1024

st.set_page_config(page_title="B&W Photo Colorizer", page_icon="🎨", layout="wide")
st.title("🎨 Black & White Photo Colorizer")
st.caption("A U-Net trained from scratch in PyTorch. It predicts the a/b colour channels of Lab colour space from the grayscale (L) channel.")


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(WEIGHTS, device), device


if not Path(WEIGHTS).exists():
    st.error("No trained model found. Run `python train.py` first (see README).")
    st.stop()

model, device = get_model()
file = st.file_uploader("Upload a black & white (or any) photo", type=["jpg", "jpeg", "png", "webp"])

if file:
    img = Image.open(file).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    with st.spinner("Colorizing..."):
        out = colorize_image(model, img, device)
    c1, c2 = st.columns(2)
    c1.subheader("Original"); c1.image(img, use_container_width=True)
    c2.subheader("Colorized"); c2.image(out, use_container_width=True)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    st.download_button("⬇️ Download colorized image", buf.getvalue(), "colorized.png", "image/png")
else:
    st.info("Upload an image to get started. Works best on photos similar to the training data.")
