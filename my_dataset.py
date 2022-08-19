from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, OD_info_dict: dict, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.OD_info_dict = OD_info_dict

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        
        tmp_path = self.images_path[item]
        others, pic_name = tmp_path.rsplit('\\', 1)
        OD_info = self.OD_info_dict[pic_name]
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        if len(OD_info) != 7:
            print('OD_info error!')

        return img, OD_info, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, OD_infos, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        OD_infos = torch.as_tensor(OD_infos)
        labels = torch.as_tensor(labels)
        return images, OD_infos, labels

