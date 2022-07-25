import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import json

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]




class LostAndFoundWithDistanceDataset(data.Dataset):

    def __init__(self, root, split='train', image_transform=None, target_transform=None, joint_transform=None):
        self.root = root
        self.images_base = os.path.join(self.root, 'leftImg8bit', split)
        self.annotations_base = os.path.join(self.root, 'gtCoarse', split)
        self.disparity_base = os.path.join(self.root, 'disparity', split)
        self.camera_base = os.path.join(self.root, 'camera', split)
        # self.files = recursive_glob(rootdir=self.annotations_base, suffix='_labelTrainIds.png')
        self.files = recursive_glob(rootdir=self.annotations_base, suffix='_gtCoarse_labelTrainIds.png')
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        print('WARNING: target transforms have no effect')

        if len(self.files) == 0:
            raise Exception("> No files found in %s" % (self.images_base))

        print("> Found %d images..." % len(self.files))

    def __len__(self):
        return len(self.files)

    def _compute_distance(self, disparity_map, camera_params):
        b = camera_params['extrinsic']['baseline']
        f = (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy'])/2.
        distance = (b * f) / disparity_map
        return distance


    def __getitem__(self, index):
        lbl_path = self.files[index].rstrip()
        img_path = lbl_path.replace('_gtCoarse_labelTrainIds.png', '_leftImg8bit.png')
        img_path = img_path.replace('gtCoarse', 'leftImg8bit')

        street, name = lbl_path.split('/')[-2:]
        name = name.replace('_gtCoarse_labelTrainIds.png', '')
        disparity_path = os.path.join(self.disparity_base, street, name + '_disparity.png')
        camera_path = os.path.join(self.camera_base, street, name + '_camera.json')


        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        img = Image.open(img_path)
        img = self.image_transform(img)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        lbl = Image.open(lbl_path)
        lbl_np = np.array(lbl)
        lbl = torch.from_numpy(lbl_np)
        lbl[lbl == 255] = 3
        lbl = lbl - 1

        # lbl = self.target_transform(lbl)

        img, lbl = self.joint_transform((img, lbl)) if self.joint_transform else (img, lbl)

        if not os.path.isfile(disparity_path) or not os.path.exists(disparity_path):
            raise Exception("{} is not a file, can not open with imread.".format(disparity_path))
        disparity = Image.open(disparity_path)
        disparity = np.array(disparity) / 256.
        disparity = torch.from_numpy(disparity)

        if not os.path.isfile(camera_path) or not os.path.exists(camera_path):
            raise Exception("{} is not a file, can not open with imread.".format(camera_path))
        with open(camera_path, 'r') as f:
            camera_params = json.load(f)

        distance = self._compute_distance(disparity, camera_params)
        max_dist = distance[~torch.isinf(distance)].max()
        distance[torch.isinf(distance)] = max_dist

        return img, lbl.unsqueeze(0), distance.unsqueeze(0)

