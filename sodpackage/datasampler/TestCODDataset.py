from torchvision import transforms
from PIL import Image

import os

from .CODDataset import CODDataset

class TestCODDataset(CODDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.image_transform = transforms.Compose([
                transforms.Resize(train_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize(train_size),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        image_main_name = os.path.splitext(os.path.basename(image_path))[0]

        return image, mask, mask_path, image_main_name