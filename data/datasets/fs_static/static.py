import json
import numpy as np
import os
import torch
from PIL import Image
from torch.utils import data

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]

class FSStaticDataset(data.Dataset):
    n_classes = 66

    def __init__(self, root, image_transform, target_transform, joint_transform):
        self.root = root

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.images = recursive_glob(rootdir=self.root, suffix='.jpg')

        print("> Found %d images..." % (len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        lbl_path = img_path.replace("_rgb.jpg", "_labels.png")

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        image = self.image_transform(image)
        label = self.target_transform(label)
        image, label = self.joint_transform((image, label)) if self.joint_transform else (image, label)
        return image, label
