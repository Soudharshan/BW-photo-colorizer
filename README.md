# 🎨 Black & White Photo Colorizer

Deep learning mini project: a **U-Net** (PyTorch) that adds colour to grayscale photos, with a **Streamlit** web demo.

![comparison](results/comparison.png)
*Columns: grayscale input | model output | original colour. (Add your own after training.)*

## How it works
1. Convert an RGB image to **Lab** colour space: `L` = lightness (this is the grayscale image), `a`/`b` = colour.
2. The network takes `L` as input and **predicts `a` and `b`** (a regression problem, trained with L1 loss).
3. Merge the original `L` with the predicted `a`,`b` and convert back to RGB.
4. At inference time colour is predicted at 128x128, upsampled, and combined with the **full-resolution original L** so details stay sharp.

## Project structure
```
colorizer/
├── data/               # training photos (put them here)
├── model/              # saved weights (colorizer.pth)
├── results/            # loss curve, metrics, comparison image
├── samples/            # sample outputs saved during training
├── model.py            # U-Net architecture
├── dataset.py          # image folder -> (L, ab) pairs
├── utils.py            # Lab conversion + colorize function
├── download_data.py    # downloads Flowers102 dataset
├── train.py            # training loop
├── evaluate.py         # PSNR / SSIM vs grayscale baseline
├── app.py              # Streamlit demo
└── requirements.txt
```

## Setup and run
```bash
pip install -r requirements.txt

# 1. Get data (or put any folder of colour photos inside data/)
python download_data.py

# 2. Train (GPU: ~10-15 min for 30 epochs on Colab; CPU is slower, use --epochs 15)
python train.py --data_dir data/flowers --epochs 30

# 3. Evaluate (writes results/metrics.txt and results/comparison.png)
python evaluate.py --data_dir data/flowers

# 4. Launch the web app
streamlit run app.py
```
No GPU? Run steps 1 to 3 on **Google Colab** (free T4), download `model/colorizer.pth`, then run the app locally.

## Tips for better results
- Train on **one category** (flowers, landscapes, animals). Narrow data gives much better colours.
- More images and more epochs help. 5,000+ images and 30 to 50 epochs is a good target.
- Outputs look slightly muted. That is normal for L1 regression, since the model plays safe when colour is ambiguous.

## Results
Fill this in after running `evaluate.py`:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Grayscale baseline | | |
| U-Net colorizer | | |

## Possible extensions
- Use a GAN loss (pix2pix style) for more vivid colours
- Train on a bigger dataset (Places365, ImageNet subset) or at 256x256
- Add a perceptual loss (VGG features)
- Deploy on Streamlit Community Cloud or Hugging Face Spaces
