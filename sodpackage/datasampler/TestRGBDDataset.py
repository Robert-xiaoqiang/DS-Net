from torchvision import transforms
import numpy as np
from PIL import Image

import os

from .transforms.JointTransformer import JointResize
from .RGBDDataset import RGBDDataset

class TestRGBDDataset(RGBDDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)

        self.joint_transform = JointResize(train_size)

        # self.data_list = self._make_list(dataset_root, [ 'RGB', 'generated_depth', 'GT' ])
        # self.mean = np.array([0.447, 0.407, 0.386])
        # self.std = np.array([0.244, 0.250, 0.253])

        # self.joint_transform = JointResize(train_size)

        # self.image_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(self.mean, self.std)
        #     ])
        # self.depth_transform = transforms.ToTensor()
        # self.mask_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image_path, depth_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image, depth, mask = self.joint_transform(image, depth, mask)
        image = self.image_transform(image)
        depth = self.depth_transform(depth)
        mask = self.mask_transform(mask)

        image_main_file_name = os.path.splitext(os.path.basename(image_path))[0]

        return image, depth, mask, mask_path, image_main_file_name
