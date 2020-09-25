from torchvision import transforms
from PIL import Image
import numpy as np

import os

from .PreTrainingDataset import PreTrainingDataset

class TestPreTrainingDataset(PreTrainingDataset):
    def __init__(self, dataset_root, train_size):
        # super().__init__(dataset_root, train_size)

        # self.joint_transform = JointResize(train_size)
        # here in Python, not C++, is-a relationship is up to ourselves

        self.data_list = self._make_list(dataset_root, [ 'RGB' ])
        self.mean = np.array([0.447, 0.407, 0.386])
        self.std = np.array([0.244, 0.250, 0.253])

        self.image_transform = transforms.Compose([
                transforms.Resize(train_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

    def __getitem__(self, index):
        image_path = self.data_list[index][0]

        image = Image.open(image_path).convert('RGB')

        image = self.image_transform(image)

        image_main_file_name = os.path.splitext(os.path.basename(image_path))[0]

        return image, image_main_file_name