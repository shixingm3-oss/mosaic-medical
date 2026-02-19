"""
Data augmentation for 2D and 3D medical images.

2D pipeline:
  - Student: ToTensor -> Normalize -> HFlip -> Rotation
  - Teacher: ToTensor -> Normalize  (no random augmentation)

3D pipeline (intensity-only — spatial structure is preserved):
  - Student: Normalize -> RandomIntensity (brightness + contrast)
  - Teacher: Normalize
"""

import random
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple


# ---- 2D transforms -----------------------------------------------------------

class Transforms2D:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    @classmethod
    def get_train_transform(cls, for_teacher=False):
        norm = transforms.Normalize(mean=cls.MEAN, std=cls.STD)
        if for_teacher:
            return transforms.Compose([transforms.ToTensor(), norm])
        return transforms.Compose([
            transforms.ToTensor(), norm,
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
        ])

    @classmethod
    def get_eval_transform(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.MEAN, std=cls.STD),
        ])


# ---- 3D transforms -----------------------------------------------------------

class Normalize3D:
    """Scale to [0, 1] and ensure shape [C, D, H, W]."""

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.float()
        if x.max() > 1:
            x = x / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x


class RandomIntensity3D:
    """Random brightness / contrast shift (no spatial deformation)."""

    def __init__(self, brightness=(-0.1, 0.1), contrast=(0.9, 1.1), p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = x + random.uniform(*self.brightness)
        if random.random() < self.p:
            m = x.mean()
            x = (x - m) * random.uniform(*self.contrast) + m
        return torch.clamp(x, 0, 1)


class Compose3D:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, x):
        for t in self.ts:
            x = t(x)
        return x


class Transforms3D:
    @classmethod
    def get_train_transform(cls, for_teacher=False):
        if for_teacher:
            return Normalize3D()
        return Compose3D([Normalize3D(), RandomIntensity3D()])

    @classmethod
    def get_eval_transform(cls):
        return Normalize3D()


# ---- Dual-augmentation wrappers (teacher-student) ----------------------------

class DualTransform2D:
    def __init__(self):
        self.student = Transforms2D.get_train_transform(for_teacher=False)
        self.teacher = Transforms2D.get_train_transform(for_teacher=True)

    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.student(img), self.teacher(img)


class DualTransform3D:
    def __init__(self):
        self.student = Transforms3D.get_train_transform(for_teacher=False)
        self.teacher = Transforms3D.get_train_transform(for_teacher=True)

    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.student(img), self.teacher(img)


def get_transforms(is_3d, split="train", dual=False):
    if split == "train":
        if dual:
            return DualTransform3D() if is_3d else DualTransform2D()
        T = Transforms3D if is_3d else Transforms2D
        return T.get_train_transform(for_teacher=False)
    T = Transforms3D if is_3d else Transforms2D
    return T.get_eval_transform()