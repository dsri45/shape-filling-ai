# Shape Filling AI

This project implements a **shape-filling neural network** that learns to convert **outline-only images of shapes** into their **filled-in counterparts**. The task is formulated as a **binary image-to-image segmentation problem** and is solved using a **custom U-Net–style convolutional neural network** implemented in **PyTorch**.

The project was designed as a lightweight vision-mask challenge, focusing on:

* Synthetic data generation
* End-to-end model training
* Inference and visualization
* Clean, minimal ML system design

---

## Project Overview

Given an input image containing **only the outline of a shape**, the model predicts a **filled mask** of that same shape.

Supported shapes include:

* Circles
* Rectangles / squares
* Ellipses
* Stars
* Small dots (already filled)
* Lines / open shapes

The model is intentionally trained on **randomly positioned, rotated, and scaled shapes** to encourage generalization.

---

## Directory Structure

```
shape-filling-ai/
│
├── data/
│   ├── train/
│   │   ├── outlines/     # Input images (shape outlines)
│   │   └── filled/       # Target images (filled shapes)
│   └── test/
│       ├── outlines/
│       └── filled/
│
├── scripts/
│   ├── generate-shapes.py  # Synthetic dataset generation
│   ├── train.py            # Model training script
│   └── predict.py          # Inference + visualization
│
├── unet.pth                # Saved trained model weights
├── sample_prediction.png   # Example model output
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Requirements

Python 3.8+

Install dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
torch
torchvision
numpy
opencv-python
```

---

## Step 1: Generate the Dataset

Synthetic training data is generated using **only NumPy and OpenCV**. Each sample consists of:

* An **outline image** (input)
* A **filled mask image** (target)

Run the dataset generation script:

```bash
python scripts/generate-shapes.py --output data --split train --count 2000
python scripts/generate-shapes.py --output data --split test --count 400
```

Arguments:

* `--output`: Root data directory
* `--split`: `train` or `test`
* `--count`: Number of images to generate
* `--size`: Image resolution (default: 256×256)

All images are saved as **single-channel PNGs** with values in `{0, 255}`.

---

## Step 2: Train the Model

The model is a **custom U-Net–style encoder–decoder** with:

* Skip connections
* Downsampling via max-pooling
* Upsampling via transposed convolutions

The task is trained as a **binary segmentation problem**.

Start training with:

```bash
cd scripts
python train.py
```

Training details:

* Loss: `Binary Cross Entropy + Dice Loss`
* Optimizer: Adam
* Epochs: 20
* Batch size: 8
* Device: GPU if available, otherwise CPU

After training, model weights are saved as:

```
unet.pth
```

---

## Step 3: Run Inference

The prediction script loads the trained model and runs inference on outline images.

Run:

```bash
python predict.py
```

This script:

1. Loads the trained U-Net
2. Selects an outline image from the test set
3. Predicts the filled mask
4. Saves a side-by-side comparison as:

```
sample_prediction.png
```

---

## Example Output

Left: Input outline
Right: Model-predicted filled shape

The model learns to:

* Close open contours
* Infer shape interiors
* Handle partial or noisy outlines

---

## Design Choices & Rationale

* **Synthetic data**: Enables unlimited labeled data with full control over shape diversity
* **U-Net architecture**: Well-suited for pixel-aligned segmentation tasks
* **Dice loss**: Encourages correct region filling even for small shapes
* **Minimal dependencies**: Improves reproducibility and portability
* **Grayscale images**: Simplifies the learning problem

---

## Limitations & Future Work

Possible extensions include:

* Multi-shape images
* Thicker or broken outlines
* Uncertainty estimation (e.g., Monte Carlo Dropout)
* Multi-class shape prediction
* More expressive architectures

---

## Environment

This project is compatible with:

* Local Python environments
* Google Colab (GPU supported)

No external datasets are required.

---

## Author

Developed as part of a vision-mask challenge focused on efficient and scalable ML architectures.

---

## License

This project is intended for educational and research purposes.
