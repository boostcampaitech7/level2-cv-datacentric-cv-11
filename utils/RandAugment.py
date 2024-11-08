# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL
import albumentations as A
import numpy as np
import torch
from PIL import Image

def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    style_transform = [
        A.RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=20, p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.Posterize(num_bits=4, p=1.0),
        A.Solarize(threshold=128, p=1.0),
        A.Equalize(p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        A.FancyPCA(alpha=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(mean = 1, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(1 - 1.0, 1 + 1.0), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(1 - 1.0, 1 + 1.0), contrast_limit=0, p=1.0),
        A.ColorJitter( p=1.0)
    ]

    return style_transform

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        #self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        transform = A.Compose(ops)
        img = np.array(img)
        
        transformed = transform(image = img)
        img =transformed['image']

        return img