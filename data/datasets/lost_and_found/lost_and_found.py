import os
import numpy as np
from torch.utils import data
from PIL import Image


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]




class LostAndFoundDataset(data.Dataset):

    def __init__(self, root, split='train', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.images_base = os.path.join(self.root, 'leftImg8bit', split)
        self.annotations_base = os.path.join(self.root, 'gtCoarse', split)
        self.files = recursive_glob(rootdir=self.annotations_base, suffix='_labelTrainIds.png')
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        if len(self.files) == 0:
            raise Exception("> No files found in %s" % (self.images_base))

        print("> Found %d images..." % len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        lbl_path = self.files[index].rstrip()
        img_path = lbl_path.replace('_gtCoarse_labelTrainIds.png', '_leftImg8bit.png')
        img_path = img_path.replace('gtCoarse', 'leftImg8bit')

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        img = Image.open(img_path)
        img = self.image_transform(img)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        lbl = Image.open(lbl_path)
        lbl = self.target_transform(lbl)

        img, lbl = self.joint_transform((img, lbl)) if self.joint_transform else (img, lbl)
        return img, lbl

