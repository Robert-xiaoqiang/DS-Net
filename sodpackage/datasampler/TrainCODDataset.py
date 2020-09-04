from .CODDataset import CODDataset
from PIL import Image

class TrainCODDataset(CODDataset):
    def __init__(self, dataset_root, train_size):
        super().__init__(dataset_root, train_size)
    
    def __getitem__(self, index):
        image_path, mask_path = self.data_list[index]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self.joint_transform(image, mask)
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask  