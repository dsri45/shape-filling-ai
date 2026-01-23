import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

# use GPU if available, otherwise fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Custom dataset for shapes
class ShapeDataset(Dataset):
    def __init__(self, root_dir):
        # List all outline files and sort them so images/masks match
        self.outline_files = sorted(os.listdir(os.path.join(root_dir, "outlines")))
        self.root = root_dir

    def __len__(self):
        return len(self.outline_files)

    def __getitem__(self, idx):
       
        filename = self.outline_files[idx]

        outline_img = cv2.imread(os.path.join(self.root, "outlines", filename), 0).astype(np.float32) / 255.0
        filled_img = cv2.imread(os.path.join(self.root, "filled", filename), 0).astype(np.float32) / 255.0

        # Convert to PyTorch tensors and add channel dimension
        outline_tensor = torch.from_numpy(outline_img).unsqueeze(0)
        filled_tensor = torch.from_numpy(filled_img).unsqueeze(0)

        return outline_tensor, filled_tensor

class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU()
            )

        # Encoder (downsampling)
        self.enc1 = conv_block(1, 32)
        self.enc2 = conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Middle block (bottleneck)
        self.bottleneck = conv_block(64, 128)

        # Decoder (upsampling)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        # Output layer: single channel mask
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder with skip connections
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Output sigmoid mask
        return torch.sigmoid(self.out_conv(d1))

# Dice loss to improve overlap accuracy for small shapes
def dice_loss(pred_mask, true_mask):
    smooth = 1e-6
    intersection = (pred_mask * true_mask).sum()
    return 1 - (2 * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

# Training function
def train_model():

    train_dataset = ShapeDataset("../data/train")
    val_dataset = ShapeDataset("../data/test")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    #  model initalization
    model = UNetModel().to(DEVICE)

    # Optimizer, loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bce_loss = nn.BCELoss()  # standard pixel-wise classification

    num_epochs = 20  

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for outlines, filled in train_loader:
            outlines = outlines.to(DEVICE)
            filled = filled.to(DEVICE)

            # Forward pass
            preds = model(outlines)

            # Combined loss: BCE + Dice
            loss = bce_loss(preds, filled) + dice_loss(preds, filled)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "unet.pth")
    print("Training complete")

# Run training
if __name__ == "__main__":
    train_model()
