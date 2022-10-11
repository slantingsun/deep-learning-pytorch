import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet


def main():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open("1.jpg")
    img = transform(img)  # [C,H,W]
    # unsqueeze增加一维度，pytorch数据的格式是[batch,channel,height,weidth]，太抽象想不出来的话，就别想了，用[B,C,H,W]代替
    img = torch.unsqueeze(img, dim=0)  # [B,C,H,W]

    save_path = './model.pth'
    model = LeNet()
    model = model.load_state_dict(torch.load(save_path))

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    print(classes[int(preds)])


if __name__ == "__main__":
    main()
