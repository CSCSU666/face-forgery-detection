from albumentations.pytorch.transforms import ToTensorV2
import albumentations as alb


def get_transforms(mode, input_size):
    if mode == 'train':
        return alb.Compose([
            # alb.Rotate(limit=30),
            # alb.CoarseDropout(1, 25, 25, p=0.1),
            # alb.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
            alb.Resize(input_size, input_size),
            alb.HorizontalFlip(),
            # alb.ToGray(p=0.1),
            # alb.GaussNoise(p=0.1),
            alb.OneOf([
                alb.RandomBrightnessContrast(),
                # alb.FancyPCA(),
                # alb.HueSaturationValue(),
            ], p=0.5),
            # alb.GaussianBlur(blur_limit=(3, 7), p=0.05),
            alb.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ToTensorV2(),
        ])
    else:
        return alb.Compose([
            alb.Resize(input_size, input_size),
            alb.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ToTensorV2(),
        ])


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(image=x)['image']
        k = self.base_transform(image=x)['image']
        return [q, k]
