import matplotlib.pyplot as plt
from torchvision import transforms
import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

def get_image_transform(IMG_SIZE):
    # weak augmentation : RandomCrop, Random Horizontal/Vertical Flip
    data_transform = transforms.Compose(
        [
            transforms.Lambda(PIL.Image.open),
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(IMG_SIZE+28,IMG_SIZE+28),antialias=True),
            transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return data_transform
def get_image_strong_augment_nocolor_transform(IMG_SIZE, num_aug_operations=6):
    # additional RandAugment
    data_transform = transforms.Compose(
        [
            transforms.Lambda(PIL.Image.open),
            # aggressive augmentation start
            transforms.RandomInvert(),
            transforms.RandomAffine(degrees=45,scale=(0.7,1.3),shear=(0.75,1.5)),
            # transforms.RandomErasing(),
            # aggressive augmentation end
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(IMG_SIZE+28,IMG_SIZE+28),antialias=True),
            transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return data_transform    

def get_image_strong_augment_transform(IMG_SIZE,num_aug_operations = 6):
    # additional RandAugment
    data_transform = transforms.Compose(
        [
            transforms.Lambda(PIL.Image.open),
            transforms.RandAugment(num_ops=num_aug_operations),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(IMG_SIZE+28,IMG_SIZE+28),antialias=True),
            transforms.RandomCrop(size=(IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return data_transform




class AutoContrast:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return PIL.ImageOps.autocontrast(img)


class Brightness:
    def __init__(self, v):
        self.v = v
        assert v >= 0.0

    def __call__(self, img):
        return PIL.ImageEnhance.Brightness(img).enhance(self.v)


class Color:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return PIL.ImageEnhance.Color(img).enhance(self.v)


class Contrast:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return PIL.ImageEnhance.Contrast(img).enhance(self.v)


class Equalize:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return PIL.ImageOps.equalize(img)


class Invert:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        return PIL.ImageOps.invert(img)


class Posterize:
    def __init__(self, v):
        self.v = int(v)
        self.v = max(1, self.v)

    def __call__(self, img):
        return PIL.ImageOps.posterize(img, self.v)


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Posterize, 4, 8),
    ]
    return l


class RandAugment:
    def __init__(self, n) -> None:
        """n: how many different operations should be apply?"""
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        print(ops)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            op_concrete = op(val)
            img = op_concrete(img)
        return img


if __name__ == "__main__":
    sample_img_path = (
        "isic_challenge/ISBI2016_ISIC_Part3B_Training_Data/ISIC_0000000.jpg"
    )

    img = PIL.Image.open(sample_img_path)
    randaug = transforms.RandAugment(6)  # apply one random augmentation
    aug_img = randaug(img)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(aug_img)
    plt.show()

    weak_augment = get_image_transform(224)
    strong_augment = get_image_strong_augment_transform(224)

    weak_augment_img = weak_augment(sample_img_path)
    strong_augment_img = strong_augment(sample_img_path)

    transforms.ToPILImage()(weak_augment_img).show()
    transforms.ToPILImage()(strong_augment_img).show()