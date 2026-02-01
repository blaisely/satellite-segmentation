import cv2
import torch

from utils import Utils

class LandCoverDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            df,
            class_rgb_values = None, 
            augmentation = None, 
            preprocessing = None,
            test = False
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.test = test
    
    def __getitem__(self, i):
        
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

        if self.test:
            if self.augmentation:
                sample = self.augmentation(image = image)
                image = sample['image']
            
            if self.preprocessing:
                sample = self.preprocessing(image = image)
                image= sample['image']

            return image

        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = Utils.reverse_one_hot(mask, self.class_rgb_values)
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_classes(self) -> list[str]:
        return ['urban_land', 'agriculture_land', 'rangeland',
                 'forest_land', 'water', 'barren_land', 'unknown']
    
    def get_rgb(self) -> list[float]:
        return self.class_rgb_values
