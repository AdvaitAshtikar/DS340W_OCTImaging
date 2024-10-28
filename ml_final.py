# Run the following command in your terminal to install the required environment variables
# pip install torch torchvision opencv-python albumentations efficientnet_pytorch kagglehub pandas scikit-learn matplotlib json

# importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import cv2
import json
import numpy as np
import pandas as pd
import os
from PIL import Image
import kagglehub

# data preparation 
# pytorch dataset for lunding the G1020 data

class GlaucomaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Read the image using OpenCV
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Corrected the typo here

        # Apply any transformations if specified
        if self.transform:
            image = self.transform(image)

        # Get the label and convert it to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert from OpenCV image to PIL image
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 1: Download the dataset using kagglehub
path = kagglehub.dataset_download("arnavjain1/glaucoma-datasets")

# Step 2: Set up paths and load CSV
image_dir = os.path.join(path, 'G1020', 'Images')  # Adjust this to the correct image directory
csv_file = os.path.join(path, 'G1020', 'G1020.csv')

# Load the CSV file containing image names and labels
df = pd.read_csv(csv_file)
image_paths = [os.path.join(image_dir, img_name) for img_name in df['imageID']] 
labels = df['binaryLabels'].values 

# Step 3: Create the dataset and dataloader
dataset = GlaucomaDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Testing: Load a batch of images
images, labels = next(iter(dataloader))
print(images.shape, labels.shape)  # Should print torch.Size([16, 3, 256, 256]) and torch.Size([16])

# building UNet model (deep learning) for segmentation of the images

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # encoding
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # decoding
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # encoding
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # decoding
        d4 = self.upconv4(e4)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)
        return torch.sigmoid(self.conv_last(d2))

unet_model = UNet()

efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')
num_features = efficientnet_model._fc.in_features
efficientnet_model._fc = nn.Linear(num_features, 1)  # Binary classification (glaucoma vs non-glaucoma)

# Training loop for UNet (Segmentation)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = unet_model.to(device)
criterion_seg = nn.BCEWithLogitsLoss()
optimizer_seg = optim.Adam(unet_model.parameters(), lr=0.001)

for epoch in range(1):
    unet_model.train()
    running_loss = 0.0
    print(f"Starting epoch {epoch + 1} for U-Net...")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Reshape labels
        labels = labels.view(-1, 1, 1, 1)  # Reshape labels to [batch_size, 1, 1, 1]
        labels = labels.expand(-1, -1, 256, 256)  # Expand to [batch_size, 1, 256, 256]

        # Forward pass
        outputs = unet_model(images)
        loss = criterion_seg(outputs, labels)
        
        # Backward pass
        optimizer_seg.zero_grad()
        loss.backward()
        optimizer_seg.step()
        
        running_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/10], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} completed. Average Loss: {running_loss / len(dataloader):.4f}\n")


from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_classification_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(all_labels, all_preds)
    print(f"AUC-ROC: {auc}")

evaluate_classification_model(efficientnet_model, dataloader)

import time

def calculate_cdr(disc_mask, cup_mask):

    disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(disc_contours) > 0:
        disc_cnt = max(disc_contours, key=cv2.contourArea)
        _, _, _, disc_height = cv2.boundingRect(disc_cnt)
    else:
        disc_height = 1  # Avoid division by zero

    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cup_contours) > 0:
        cup_cnt = max(cup_contours, key=cv2.contourArea)
        _, _, _, cup_height = cv2.boundingRect(cup_cnt)
    else:
        cup_height = 0

    cdr = cup_height / disc_height
    return cdr
unet_model.eval()
cdr_list = []

with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        start_time = time.time()
        output_masks = unet_model(images)
        inference_time = time.time() - start_time

        for i in range(images.size(0)):
            output_mask = output_masks[i].cpu().numpy().squeeze()
            output_mask = (output_mask > 0.5).astype(np.uint8)

            # Assume disc_mask and cup_mask separation logic here
            disc_mask = output_mask  # Modify this based on your actual model output
            cup_mask = output_mask    # Modify this based on your actual model output

            # Calculate CDR
            cdr = calculate_cdr(disc_mask, cup_mask)
            cdr_list.append(cdr)

print(cdr_list)  # Output the CDRs for verification

