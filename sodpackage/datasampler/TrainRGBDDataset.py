from .RGBDDataset import RGBDDataset
from PIL import Image

class TrainRGBDDataset(RGBDDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)
    
    def __getitem__(self, index):
        image_path, depth_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image, depth, mask = self.joint_transform(image, depth, mask)
        image = self.image_transform(image)
        depth = self.depth_transform(depth)
        mask = self.mask_transform(mask)

        return image, depth, mask