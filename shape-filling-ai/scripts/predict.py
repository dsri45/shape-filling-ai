import os
import cv2
import torch
from train import UNetModel, DEVICE


# Load the trained model
model = UNetModel().to(DEVICE)
model.load_state_dict(torch.load("unet.pth", map_location=DEVICE))
model.eval()

# Choose an outline image to predict
folder = "../data/test/outlines"
files = sorted(os.listdir(folder))
file = files[0]  # first image


# Load and preprocess the image
img = cv2.imread(os.path.join(folder,file),0)
x = torch.from_numpy(img.astype(float)/255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

# Run the model
with torch.no_grad():
    pred = model(x)[0,0].cpu().numpy()

# Stack outline + prediction
combined = cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor((pred*255).astype('uint8'), cv2.COLOR_GRAY2BGR)])

# Save the combined image
cv2.imwrite("sample_prediction.png", combined)

