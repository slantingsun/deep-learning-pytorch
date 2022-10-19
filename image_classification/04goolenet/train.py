import json
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from googlenet import GoogLeNet
from tqdm import tqdm
import sys


# import matplotlib.pyplot as plt


def main():
    # 设置data的model的device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    image_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    data_root = r'./data'
    assert os.path.exists(data_root), f"{data_root} path not exist."

    batch_size = 64
    # 设置num_workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    # 使用ImageFolder构成dataset
    # 文件结构
    #  train
    #   - daisy
    #   - dandelion
    #   ......
    #   - tulips
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

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

    model = GoogLeNet(num_classes=10, init_weights=True)

    # *********************新增*********************

    # 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
    # 官方的模型中使用了bn层以及改了一些参数，不能混用
    # import torchvision
    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)

    # 流水线三件套
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

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
