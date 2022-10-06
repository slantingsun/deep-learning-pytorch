import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet
from tqdm import tqdm
import sys


def main():
    data_dir = r'./data'
    image_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    train_dataset = datasets.CIFAR10(data_dir, True, image_transforms['train'], download=True)
    val_dataset = datasets.CIFAR10(data_dir, False, image_transforms['val'], download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # model.fc = nn.Linear(model.fc.in_features, 10)
    model = LeNet()
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    epoch = 5
    for i in range(epoch):
        print(f"Epoch {i + 1}/{epoch}")
        running_loss = 0.0
        running_corrects = 0.0
        model.train()
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += ((preds == labels).sum().item())
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_corrects / len(train_dataset)

        running_loss = 0.0
        running_corrects = 0.0
        with torch.no_grad():
            val_iter = tqdm(val_dataloader, file=sys.stdout)
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.eval()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += ((preds == labels).sum().item())
        epoch_val_loss = running_loss / len(val_dataset)
        epoch_val_acc = running_corrects / len(val_dataset)
        print(f"Train Loss {epoch_train_loss} Acc{epoch_train_acc}")
        print(f"Val Loss {epoch_val_loss} Acc{epoch_val_acc}")


if __name__ == '__main__':
    main()
