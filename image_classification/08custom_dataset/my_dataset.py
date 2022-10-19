from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 为什么使用PIL的Image读取图片，而不是cv2
        # 因为pytorch提供的transform方法都是针对Image格式
        # cv2的imshow读取方式同imread的色彩通道顺序，默认BGR
        # PIL读入方式为RGB,plt显示图片默认为RGB
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        # 这里可以自定义transform方法，不使用官方的，有能力的话，参考yolov3
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        # stack会先增加一个维度，再做拼接
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
