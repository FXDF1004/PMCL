'''
@contact:xind2023@mail.ustc.edu.cn
@time:2025/9/1
'''

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

normalize =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

color_transforms = [
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
]

shape_transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
]

augmentation_rand =transforms.Compose( [
    transforms.RandomApply(color_transforms, p=1.0),

    transforms.RandomApply(shape_transforms, p=1.0),

    transforms.RandomResizedCrop(224),

    transforms.ToTensor(),

    normalize,
]
)

augmentation_sim = transforms.Compose(
    [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    
    normalize,
    ]
    # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
    )


augmentation_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))

        normalize,
    ])

class isic2019_dataset(Dataset):
    def __init__(self,path,transform,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_train5.csv'))
            # self.df = pd.read_csv(os.path.join(path,'train.txt'))
            # txt_train = f'data/ISIC2019/train.txt'
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_test5.csv'))
            # self.df = pd.read_csv(os.path.join(path,'test.txt'))
            # txt_val = f'data/ISIC2019/test.txt'
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_test5.csv'))
            # self.df = pd.read_csv(os.path.join(path,'test.txt'))
            # txt_test = f'data/ISIC2019/test.txt'

    def __getitem__(self, item):
        # img_path = os.path.join(self.path,'ISIC2019_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img_path = os.path.join('', f"{self.df.iloc[item]['image']}")
        img = Image.open(img_path)
        try:
            img = Image.open(img_path)
            # print(f"Loading image from: {img_path}")
            if img.mode != 'RGB':
                img = img.convert("RGB")
        except (OSError, IOError) as e:
            print(f"Error loading image: {img_path}, error: {e}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])
        

        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)

                return [img1,img2],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))


class isic2018_dataset(Dataset):
    def __init__(self,path,transform,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_train5.csv'))
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_test5.csv'))
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_test5.csv'))

    def __getitem__(self, item):
        # img_path = os.path.join(self.path,'ISIC2018_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img_path = os.path.join('', f"{self.df.iloc[item]['image']}")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])

        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)

                return [img1,img2],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))

