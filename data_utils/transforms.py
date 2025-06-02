import math
import random
import numbers
from numpy.core.fromnumeric import size
import torch
import numpy as np
from torchvision.transforms import Compose


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, heat):
        for t in self.transforms:
            points, heat = t(points, heat)
        return points, heat


class NormalizeUnitSphere:
    def __call__(self, points, heat):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points = points / scale
        heat = heat / scale
        return points, heat
    


class AddNoise:
    def __init__(self, noise_std_min=0.0, noise_std_max=0.02):
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, points, heat):
        std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.normal(scale=std, size=points.shape)
        return points + noise, heat


class RandomScale:
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, points, heat):
        scale = np.random.uniform(*self.scale_range)
        return points * scale, heat * scale


class RandomRotate:
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, points, heat):
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        if self.axis == 0:
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s, c]])
        elif self.axis == 1:
            R = np.array([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]])
        else:  # axis == 2
            R = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]])
        return points @ R.T, heat


def standard_train_transforms(noise_std_min=0.0, noise_std_max=0.01, scale_d=0.2, rotate=True):
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        RandomScale([1.0 - scale_d, 1.0 + scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)