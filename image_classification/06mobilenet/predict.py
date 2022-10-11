import json
import os.path

import numpy
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

    img_path = "./1.jpg"
    assert os.path.exists(img_path), f"path {img_path} is not exist."
    img = Image.open(img_path)
    # plt.imshow(img)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = "./class_idx.json"
    assert os.path.exists(json_path), f"path {json_path} is not exist."
    with open("./class_idx.json", "r") as json_file:
        class_idx = json.load(json_file)

    model = MobileNetV2(num_classes=5).to(device)
    weight_path = "./AlexNet.pth"
    assert os.path.exists(weight_path), f"path {weight_path} is not exist."

    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        outputs = model(img.to(device))
        outputs = torch.squeeze(outputs).cpu()
        outputs = torch.softmax(outputs, dim=0)
        # torch.argmax 返回输入张量中指定维度的最大值的索引。
        # torch.max(x , dim=1)返回两个结果 第一个是最大值，第二个是对应的索引值
        preds_idx = torch.argmax(outputs)

    plt_title = f"class:{class_idx[str[preds_idx]]} probability:{outputs[preds_idx].numpy():.3f}"
    # plt.title(plt_title)
    for i in range(len(preds_idx)):
        print(f"class:{class_idx[str[i]]} probability:{preds_idx[i].numpy():.3f}")
