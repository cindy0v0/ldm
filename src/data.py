import os
import json
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from enum import Enum
from torchvision import transforms


class PathologyBase(Dataset):
    def __init__(self, size, center_crop=True, random_flip=False, jitter=False):
        super().__init__()
        # 20x magnification
        root = "/projects/ovcare/users/cindy_shi/ldm/data/ocean.json"
        with open(root, "r") as f:
            self.data = json.load(f)
        self.labels = None

        factor = 0.0
        if jitter:
            factor = 0.01
        self.transforms = transforms.Compose([
            transforms.CenterCrop(
                256) if center_crop else transforms.RandomCrop(size),
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=factor, contrast=factor, saturation=factor, hue=factor),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        label = self.labels[i]
        ex = {}
        ex["label"] = label
        ex["input"] = self.transforms(Image.open(example).convert("RGB"))
        ex["path"] = example

        return ex


class PathologyTrain(PathologyBase):
    def __init__(self, size, keys=None, paths=None, labels=None, subsample=False, n_samples=None, center_crop=True, jitter=False):
        super().__init__(size, center_crop=center_crop, random_flip=False, jitter=jitter)
        data = self.data['train']
        paths = data['images'] if paths is None else paths
        labels = [0] * len(paths) if labels is None else labels
        if subsample:
            indices = list(range(len(paths)))[:5000]
            paths = paths[indices]
            labels = labels[indices]
            self.indices = indices
        if n_samples and n_samples < len(paths):
            paths = paths[:n_samples]
            labels = labels[:n_samples]

        self.data = paths
        self.labels = labels


class PathologyValidation(PathologyBase):
    def __init__(self, size, debug=False, subsample=False, n_samples=None, seed=42):
        super().__init__(size, random_flip=False)
        data = self.data['validation']
        paths = data['images']
        labels = [0] * len(paths)

        if subsample:
            indices = list(range(len(paths)))
            indices = random.sample(indices, 50)
            paths = np.array(paths)[indices]
            labels = np.array(labels)[indices]
            self.indices = indices

        if n_samples:
            # randomly select n_samples
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), size=n_samples, replace=False)
            paths = np.array(paths)[idx]
            labels = np.array(labels)[idx]
        self.debug = debug
        self.data = paths
        self.labels = labels
        self.subsample = subsample


class PathologyTest(PathologyBase):
    def __init__(self, size, external_set):
        super().__init__(size, random_flip=False)
        data = self.data[external_set]
        paths = data['images']
        labels = data['labels']
        labels = [PathologyLabels.VALID.value[label] for label in labels]
        self.data = paths
        self.labels = labels


# recording the labels in ldm ocean_external
#
class PathologyLabels(Enum):
    # some extra metadata
    LABEL_IDX = -5
    SID_IDX = -4
    MAJOR_SUBTYPES = ['cc', 'ec', 'hgsc', 'lgsc', 'mc']
    CALIBRATION_LABEL = {'stroma': 1, 'necrosis': 2, 'tumor': 0}
    OTHER_IDX = 5  # index of other in external sets

    # maps labels in img path to labels in ldm ocean_external
    TRAIN = {'cc': 0, 'ec': 0, 'hgsc': 0, 'lgsc': 0, 'mc': 0}
    VALID = {'necrosis': 1, 'stroma': 1, 'tumor': 0}
    EXTERNAL_SET_1 = {'cc': 0, 'ec': 1,
                      'hgsc': 2, 'lgsc': 3, 'mc': 4, 'other': 5}
    EXTERNAL_SET_2 = {'cc': 0, 'ec': 1,
                      'hgsc': 2, 'lgsc': 3, 'mc': 4, 'other': 5}
