import json
import os
import time

import numpy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_v2 import MobileNetV2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def main():
    # 设置data的model的device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_root = r'./data'
    assert os.path.exists(data_root), f"{data_root} path not exist."

    batch_size = 64
    # 设置num_workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=image_transforms["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=image_transforms["val"])

    # 将文件名与标签名对应字典，写入文件
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    data_name_list = train_dataset.class_to_idx
    class_dict = {key: value for key, value in data_name_list.items()}
    # json.dumps() 是把python对象转换成json对象的一个过程，生成的是字符串。
    # json.dump() 是把python对象转换成json对象生成一个fp的文件流，和文件相关
    json_str = json.dumps(class_dict, indent=4)
    with open("./class_idx.json") as json_file:
        json_file.write(json_str)

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=nw)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # test_data_iter = iter(val_dataloader)
    # test_data = test_data_iter.next()
    # def img_show(img):
    #     img = img/2+0.5
    #     npimg = img.numpy()
    #     plt.imshow(numpy.transpose(npimg,(1,2,0)))
    #     plt.show()

    # 预训练模型
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # model.fc = nn.Linear(model.fc.in_features, 10)

    # ******************新增************************
    model = MobileNetV2(num_classes=5)
    # 官方下载
    model_weight_path = "./MobileNetV2-pre.pth"
    assert os.path.exists(model_weight_path), f"{model_weight_path} path not exist."
    pre_weights = torch.load(model_weight_path, map_location="cpu")

    # *********************新增*********************
    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
    print(f"missing_keys:{missing_keys}")
    print(f"unexpected_keys{unexpected_keys}")

    # *********************新增*********************
    for param in model.features.parameters():
        param.requires_grad = False

    # 流水线三件套
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, 0.001)

    epochs = 5
    best_acc = 0.0
    save_path = './AlexNet.pth'
    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
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

            # 设置tqdm显示信息
            train_bar.desc = f"train epochs {i + 1}/{epochs} loss:{loss:.3f}"
        # 也可以直接n+=1,直接除以n，上面去掉乘以inputs.size(0)
        epoch_train_loss = running_loss / train_num
        epoch_train_acc = running_corrects / train_num

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
        epoch_val_loss = running_loss / val_num
        epoch_val_acc = running_corrects / val_num
        print(f"Train Loss {epoch_train_loss} Acc{epoch_train_acc}")
        print(f"Val Loss {epoch_val_loss} Acc{epoch_val_acc}")
        if best_acc < epoch_val_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = epoch_val_acc


if __name__ == '__main__':
    main()
