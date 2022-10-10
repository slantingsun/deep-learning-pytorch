import time
import torch
import torch.nn as nn
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

    # 设置data的model的device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预训练模型
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # model.fc = nn.Linear(model.fc.in_features, 10)

    model = LeNet()
    # 流水线三件套
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    epoch = 5
    best_acc = 0.0
    save_path = './model.pth'
    for i in range(epoch):
        print(f"Epoch {i + 1}/{epoch}")
        # 叠加每一个batch的loss和准确率，最后求评价
        running_loss = 0.0
        running_corrects = 0.0
        model.train()
        # 给读取的可迭代对象加个进度条，很简单
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # 这里乘以inputs.size(0)，后面epoch_train_loss又除以len(train_dataset)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += ((preds == labels).sum().item())
        # 也可以直接n+=1,直接除以n，上面去掉乘以inputs.size(0)
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_corrects / len(train_dataset)

        running_loss = 0.0
        running_corrects = 0.0
        model.eval()
        with torch.no_grad():
            val_iter = tqdm(val_dataloader, file=sys.stdout)
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += ((preds == labels).sum().item())
        epoch_val_loss = running_loss / len(val_dataset)
        epoch_val_acc = running_corrects / len(val_dataset)
        print(f"Train Loss {epoch_train_loss} Acc{epoch_train_acc}")
        print(f"Val Loss {epoch_val_loss} Acc{epoch_val_acc}")
        if best_acc < epoch_val_acc:
            torch.save(model.state_dict(),save_path)
            best_acc = epoch_val_acc

if __name__ == '__main__':
    main()
