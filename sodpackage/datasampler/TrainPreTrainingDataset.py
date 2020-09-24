from .PreTrainingDataset import PreTrainingDataset
from PIL import Image

class TrainRGBDDataset(PreTrainingDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)
    
    def __getitem__(self, index):
        image_path, depth_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        image, depth = self.joint_transform(image, depth)
        image = self.image_transform(image)
        depth = self.depth_transform(depth)

        return image, depth