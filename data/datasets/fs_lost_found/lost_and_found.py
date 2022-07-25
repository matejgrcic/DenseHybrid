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




class LostFoundDataset(data.Dataset):

    def __init__(self, root, image_transform, target_transform, joint_transform):
        self.root = root
        self.images_base = os.path.join(self.root, 'leftImg8bit')
        self.annotations_base = os.path.join(self.root, 'ood')
        self.files = recursive_glob(rootdir=self.annotations_base, suffix='.png')
        self.ignore_class = 2
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
        img_path = lbl_path.replace('_ood_segmentation.png', '_leftImg8bit.png')
        img_path = img_path.replace('ood', 'leftImg8bit')

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        img = Image.open(img_path)
        img = self.image_transform(img)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        lbl = Image.open(lbl_path)
        lbl = self.target_transform(lbl)

        return self.joint_transform((img, lbl)) if self.joint_transform else (img, lbl)

