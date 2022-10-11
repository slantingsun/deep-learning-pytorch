import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from model import LeNet
import torchvision.models


def main():
    data_dir = r'../../Beginner/data'
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
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataloader = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=0) for x in
                  ['train', 'val']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    epoch = 5

    best_acc = 0.0
    best_model_wts = model.state_dict()
    since = time.time()
    # 将训练和验证放在了一起，应该每训练一个epoch，验证一次
    for i in range(epoch):
        print(f"Epoch {i + 1}/{epoch}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # lr_schedule.step()
                        # 刚开始我把lr_schedule.step()放在这里，效果很差，官网给的建议是放在val之后
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += ((preds == labels).sum().item())
            if phase == 'train':
                lr_schedule.step()
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
            print(f"{phase} Loss: {epoch_loss} Acc {epoch_acc}")
    time_ephased = time.time() - since
    print(f"Training complete in {time_ephased / 60}m {time_ephased % 60}s")
    print(f"Best val Acc is {epoch_acc:.3f}")
    save_path = "./pth/LeNet.pth"
    if not os.path.exists(save_path):
        os.mkdir(os.path.dirname())
    torch.save(best_model_wts, save_path)


if __name__ == '__main__':
    main()
