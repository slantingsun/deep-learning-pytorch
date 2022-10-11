import json
import os.path

import torch
import torch.nn as nn
from torchvision import transforms
from model import resnet34
from PIL import Image


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_root = "./data/img"
    assert os.path.exists(img_root), f"path {img_root} dose not exist."

    img_path_list = [os.path.join(img_root, i) for i in os.listdir(img_root) if i.endswith(".jpg")]

    json_path = "class_indices.json"
    assert os.path.exists(json_path), f"path {json_path} dose not exist."

    with open(json_path, "r") as f:
        class_idx = json.load(f)

    # model load weight
    model = resnet34(num_classes=5).to(device)
    weight_path = "./AlexNet.pth"
    assert os.path.exists(weight_path), f"path {weight_path} is not exist."

    # predict
    model.eval()
    batch_size = 8
    with torch.no_grad():
        for idx in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[batch_size * idx:batch_size * (idx + 1)]:
                assert os.path.exists(img_path), f"path {img_path} dose not exist."
                img = Image.open(img_path)
                img = transform(img)
                img_list.append(img)

            # 将img_list打包成一个batch
            inputs = torch.stack(img_list, dim=0)
            outputs = model(inputs.to(device)).cpu()
            predicts = torch.softmax(outputs, dim=1)
            # 获的probability和索引index
            probs, indexs = torch.max(predicts, dim=1)

            for img_index, prob, class_index in enumerate(zip(probs, indexs)):
                print(
                    f"image{img_path_list[batch_size * idx + img_index]}, class:{class_idx[class_index]}, probability:{prob}")
