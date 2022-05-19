import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# Patch extraction. (128x128x128)
class PatchExtract(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


# modified random flip. 1/3
class Random_Flip(object):
    def __call__(self, sample, prob=0.33):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.33:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.33:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.33:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}

# rotation


class Random_rotate(object):
    def __init__(self, rotate_angle, **kwargs):
        self.rotate_angle = rotate_angle
    def __call__(self, sample, **kwargs):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(0, self.rotate_angle), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


# scaling : random (10 or 20)
class Random_Scaling(object):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):  # factor in %
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(
            1.0-self.factor, 1.0+self.factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor
        # label = label*scale_factor

        return {'image': image, 'label': label}

# brightness: gamma with gain and gamma between 0.8 and 1.2 from uniform.


class Random_Brightness(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        gamma = np.random.uniform(0.8, 1.2)
        gain = np.random.uniform(0.8, 1.2)
        image = gain * image ** gamma
        label = gain * label ** gamma
        return {'image': image, 'label': label}

# elastic deformation: square with random size and random displacement. std dev of 2,5, 8,10 voxels. spline filter with order 3


class Random_Elastic(object):
    def __init__(self, transform_size):
        self.transform_size = transform_size
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        size = np.random.normal(0, self.transform_size)
        image = ndimage.interpolation.map_coordinates(
            image, self.__elastic_deformation(size, size), order=3)
        label = ndimage.interpolation.map_coordinates(
            label, self.__elastic_deformation(size, size), order=3)
        return {'image': image, 'label': label}


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(
            1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor,
                                         size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)),
                       mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    # (240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


def random_rotation(rotation_angle=90):
    return transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_rotate(rotation_angle),
        ToTensor()
    ])


def random_elastic(transform_size=5):
    return transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_Elastic(transform_size),
        ToTensor()
    ])


dict_transforms = {
    "flip+pe": transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_Flip(),
        ToTensor()
    ]),
    "bright+pe": transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_Brightness(),
        ToTensor()
    ]),
    "scale10+pe": transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_Scaling(0.1),
        ToTensor()
    ]),
    "scale20+pe": transforms.Compose([
        Pad(),
        PatchExtract(),
        Random_Scaling(0.2),
        ToTensor()
    ]),
    "rot15+pe": random_rotation(rotation_angle=15),
    "rot30+pe": random_rotation(rotation_angle=30),
    "rot60+pe": random_rotation(rotation_angle=60),
    "rot90+pe": random_rotation(rotation_angle=90),
    "elastic2+pe": random_elastic(transform_size=2),
    "elastic5+pe": random_elastic(transform_size=5),
    "elastic8+pe": random_elastic(transform_size=8),
    "elastic10+pe": random_elastic(transform_size=10),
}


# def transform(sample):
#     trans = transforms.Compose([
#         Pad(),
#         # Random_rotate(),  # time-consuming
#         Random_Crop(),
#         Random_Flip(),
#         Random_intencity_shift(),
#         ToTensor()
#     ])

#     return trans(sample)


def transform_valid(sample):
    """
    Subh: This is to be left as is. No transforms should be done on validation.
    """
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        ToTensor()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train', transforms=None, subset=None):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths
        if subset !=0:
            self.names = self.names[:subset]
            self.paths = self.paths[:subset]
        try:
            self.transforms = dict_transforms[transforms]
            print("Transforms: {}".format(transforms))
        except KeyError:
            self.transforms = None

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = self.transforms(sample)
            return sample['image'], sample['label']
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            return sample['image'], sample['label']
        else:
            image = pkload(path + 'data_f32b0.pkl')
            image = np.pad(
                image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]