import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ShapeDataset(Dataset):
    def __init__(self, root):
        self.outlines = sorted(os.listdir(os.path.join(root, "outlines")))
        self.root = root

    def __len__(self):
        return len(self.outlines)

    def __getitem__(self, idx):
        name = self.outlines[idx]
        img = cv2.imread(os.path.join(self.root, "outlines", name), 0).astype(np.float32)/255.0
        mask = cv2.imread(os.path.join(self.root, "filled", name), 0).astype(np.float32)/255.0
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )
        self.enc1 = block(1,32)
        self.enc2 = block(32,64)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(64,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = block(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,2)
        self.dec1 = block(64,32)
        self.out = nn.Conv2d(32,1,1)

    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        m=self.mid(self.pool(e2))
        d2=self.up2(m)
        d2=self.dec2(torch.cat([d2,e2],1))
        d1=self.up1(d2)
        d1=self.dec1(torch.cat([d1,e1],1))
        return torch.sigmoid(self.out(d1))

def dice_loss(pred, target):
    smooth = 1e-6
    inter = (pred*target).sum()
    return 1 - (2*inter+smooth)/(pred.sum()+target.sum()+smooth)

def train():
    train_ds = ShapeDataset("../data/train")
    val_ds = ShapeDataset("../data/test")
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for epoch in range(20):
        model.train()
        total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)
            loss = bce(p, y) + dice_loss(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1} | Train Loss: {total/len(train_loader):.4f}")

    torch.save(model.state_dict(), "unet.pth")

if __name__ == "__main__":
    train()
