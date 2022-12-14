import json
import os

import torch
from lxml import etree
from torch.utils.data import Dataset
from PIL import Image


class VOC2012DataSet(Dataset):
    def __init__(self, root, transform=None, txt_name='train.txt'):
        self.root = os.path.join(root, 'VOC2012')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, "Annotations")
        txt_path = os.path.join(self.root, 'ImageSets', 'Main', txt_name)
        assert os.path.exists(txt_path), f"not found {txt_path}"
        with open(txt_path, 'r') as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Waring:not found '{xml_path}', skip this anatotation file")
                continue
            with open(xml_path) as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data.keys():
                print(f"INFO: no objects in {xml_path}, skip this anatotation file")
                continue
            self.xml_list.append(xml_path)
        assert len(xml_list), f"found no information in {txt_path}"

        classes_json_file = "pascal_voc_classes.json"
        with open(classes_json_file, "r") as f:
            self.class_dict = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path, 'r') as read:
            xml_str = read.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError(f"Image {img_path} is not JPEG format.")

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax <= xmin or ymax <= ymin:
                print(f"Waring: in {xml_path}, there are some bbox w/h <=0")
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        """
        as_tensor?????????from_array???????????????tensor???????????????????????????????????????????????????????????????numpy array??????????????????????????????
        ?????????????????????numpy array???????????????????????????
        ???Tensor?????????tensor?????????????????????????????????????????????????????????????????????tensor??????
        """
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = [(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])]
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        height = data["size"]["height"]
        width = data["size"]["width"]
        return height, width

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # ????????????????????????
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # ??????object?????????????????????????????????????????????
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


if __name__ == '__main__':
    dataset = VOC2012DataSet('C:\personal\develop\python\dataset')
    dataset.__getitem__(1)
