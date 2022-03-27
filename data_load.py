import os
import torch
from torchvision import models, transforms, datasets
from hyper_parameters import BATCH_SIZE

data_dir = "maps"

data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root=os.path.join(
    data_dir, "train"), transform=data_transform)
dataset_val = datasets.ImageFolder(root=os.path.join(
    data_dir, "val"), transform=data_transform)

dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=24, shuffle=True, num_workers=0)
dataloader_val_big = torch.utils.data.DataLoader(
    dataset_val, batch_size=1098, shuffle=True, num_workers=0)

# print(len(dataset_train)) -> 1096
# print(len(dataset_val)) -> 1098
