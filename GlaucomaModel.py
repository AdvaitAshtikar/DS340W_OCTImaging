import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import label

class TestDataset(Dataset):
    def __init__(self, image_dir, output_size=(256, 256)):
        self.output_size = output_size
        self.image_dir = image_dir
        self.image_filenames = [path for path in os.listdir(image_dir) if not path.startswith('.')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_dir, self.image_filenames[idx])
            img = np.array(Image.open(img_name).convert('RGB'))
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.resize(img, self.output_size, interpolation=Image.BILINEAR)
            return img
        except Exception as e:
            return None

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        factor = 2
        self.down6 = Down(2048, 4096 // factor)
        self.up1 = Up(4096, 2048 // factor)
        self.up2 = Up(2048, 1024 // factor)
        self.up3 = Up(1024, 512 // factor)
        self.up4 = Up(512, 256 // factor)
        self.up5 = Up(256, 128 // factor)
        self.up6 = Up(128, 64)
        self.output_layer = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        out = self.up1(x7, x6)
        out = self.up2(out, x5)
        out = self.up3(out, x4)
        out = self.up4(out, x3)
        out = self.up5(out, x2)
        out = self.up6(out, x1)
        return torch.sigmoid(self.output_layer(out))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

EPS = 1e-7

def compute_dice_coef(input, target):
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k, :, :], target[k, :, :]) for k in range(batch_size)]) / batch_size

def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection) / (iflat.sum() + tflat.sum())

def vertical_diameter(binary_segmentation):
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)
    diameter = np.max(vertical_axis_diameter, axis=1)
    return diameter

def vertical_cup_to_disc_ratio(od, oc):
    cup_diameter = vertical_diameter(oc)
    disc_diameter = vertical_diameter(od)
    return cup_diameter / (disc_diameter + EPS)

def compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc):
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err, pred_vCDR, gt_vCDR

def refine_seg(pred):
    np_pred = pred.numpy()
    largest_ccs = []
    for i in range(np_pred.shape[0]):
        labeled, ncomponents = label(np_pred[i, :, :])
        bincounts = np.bincount(labeled.flat)[1:]
        largest_cc = labeled == np.argmax(bincounts) + 1 if len(bincounts) != 0 else labeled == 0
        largest_ccs.append(torch.tensor(largest_cc, dtype=torch.float32))
    return torch.stack(largest_ccs)

def predict_and_evaluate(model, test_loader, test_set, device, vCDR_threshold=0.6):
    results = []

    with torch.no_grad():
        for img, filename in zip(test_loader, test_set.image_filenames):
            img = img.to(device)
            logits = model(img)
            pred_od = (logits[:, 0, :, :] >= 0.5).type(torch.int8).cpu()
            pred_oc = (logits[:, 1, :, :] >= 0.5).type(torch.int8).cpu()
            pred_od_refined = refine_seg(pred_od).to(device)
            pred_oc_refined = refine_seg(pred_oc).to(device)
            try:
                pred_vCDR = vertical_cup_to_disc_ratio(pred_od_refined.cpu().numpy(), pred_oc_refined.cpu().numpy())[0]
            except Exception as e:
                pred_vCDR = None
            if pred_vCDR is not None:
                predicted_label = "Glaucoma" if pred_vCDR > vCDR_threshold else "No Glaucoma"
            else:
                predicted_label = "Error"
            results.append({
                "filename": filename,
                "vCDR": f"{pred_vCDR:.2f}" if pred_vCDR is not None else "undefined",
                "prediction": predicted_label
            })

    return results
