from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
class ImageFolder(Dataset):
    def __init__(self,image_path_list,transform) -> None:
        super().__init__()
        self.image_path_list = image_path_list
        self.transform = transform
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image=self.transform(image=image)["image"]
        image=ToTensorV2()(image=image)["image"]
        return image, image_path
    def __len__(self):
        return len(self.image_path_list)