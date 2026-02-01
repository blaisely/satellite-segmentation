import albumentations as album
from albumentations.pytorch import ToTensorV2

class Augmentor:

    @staticmethod
    def get_training_augmentation():
        train_transform = [
            album.Resize(1024, 1024),
            album.HorizontalFlip(p=0.5),
            album.VerticalFlip(p=0.5),
            album.RandomRotate90(p=0.5),
            album.OneOf([
                album.RandomBrightnessContrast(p=1),
                album.HueSaturationValue(p=1),
            ], p=0.3),
            
            album.OneOf([
                album.GaussNoise(p=1),
                album.ISONoise(p=1),
            ], p=0.2),
            
            album.GaussianBlur(blur_limit=(3, 5), p=0.1),
            album.RandomCrop(height=640, width=640, p = 1),
        ]
        return album.Compose(train_transform)

    @staticmethod
    def get_validation_augmentation():
        train_transform = [
            album.Resize(height=640, width=640, p = 1),
        ]
        return album.Compose(train_transform)

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    @staticmethod
    def get_preprocessing(preprocessing_fn=None):
        _transform = []
        if preprocessing_fn:
            _transform.append(album.Lambda(image=preprocessing_fn))
        _transform.append(ToTensorV2())
            
        return album.Compose(_transform)