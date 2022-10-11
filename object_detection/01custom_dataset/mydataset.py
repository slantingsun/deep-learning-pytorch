import torch
from torch.utils.data import dataset

"""
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""


class Mydataset():
    def __init__(self):
        pass

    def __len__(self):
        # 获取数据集的长度
        pass

    def __getitem__(self, index):
        # 返回图片以及对应的信息
        # boxes labels image_id area iscrowed
        pass

    def get_height_and_width(self):
        """
        如果调用多gpu训练，需要这个方法

        Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratios),
        then it is recommended to also implement a get_height_and_width method, which returns the height and the width of the image.
        If this method is not provided, we query all elements of the dataset via __getitem__ , which loads the image in memory and is slower than if a custom method is provided.
        """
        pass
