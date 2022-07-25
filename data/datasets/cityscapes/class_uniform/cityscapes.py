"""
Cityscapes Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms
from torch.utils import data
from torchvision.datasets import Cityscapes as _Cityscapes
import itertools

import torch
import torchvision.transforms as transforms
import data.datasets.cityscapes.class_uniform.uniform as uniform
import data.datasets.cityscapes.class_uniform.city_labels as cityscapes_labels
import copy

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
id_to_oodid = cityscapes_labels.label2oodid
num_classes = 19
ignore_label = 255
img_postfix = '_leftImg8bit.png'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

rgb_mean_std = ([0.2869, 0.3251, 0.2839], [0.1761, 0.1810, 0.1777])
normalize = transforms.Normalize(*rgb_mean_std)


class CityScapesUniform(data.Dataset):
    """
    Please do not use this for AGG
    """

    def __init__(self, dataroot, split='train', joint_transform_list=None, sliding_crop=None,
                 transform=None, target_transform=None, target_aux_transform=None, dump_images=False,
                 class_uniform_pct=0.5, class_uniform_tile=1024, coarse_boost_classes=None):
        self.joint_transform_list = joint_transform_list
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.coarse_boost_classes = coarse_boost_classes

        _cs = _Cityscapes(dataroot, split=split, mode='fine', target_type='semantic')
        targets = list(itertools.chain.from_iterable(_cs.targets))
        assert len(targets) == len(_cs.images)
        self.imgs = [(img, lbl) for img, lbl in zip(_cs.images, targets)]

        self.aug_imgs = []
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for fine data
        json_fn = 'cityscapes_{}_tile{}.json'.format('fine', self.class_uniform_tile)
        if os.path.isfile(json_fn):
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            self.centroids = {int(idx): centroids[idx] for idx in centroids}
        else:
            self.centroids = uniform.class_centroids_all(
                self.imgs,
                num_classes,
                id2trainid=id_to_trainid,
                tile_size=class_uniform_tile)
            with open(json_fn, 'w') as outfile:
                json.dump(self.centroids, outfile, indent=4)
        self.build_epoch()

    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.class_uniform_pct > 0:
            if cut:
                # after max_cu_epoch, we only fine images to fine tune
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                        self.fine_centroids,
                                                        num_classes,
                                                        0.5)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs + self.aug_imgs,
                                                        self.centroids,
                                                        num_classes,
                                                        0.5)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        seg_mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            seg_mask_copy[mask == k] = v

        ood_mask_copy = mask.copy()
        for k, v in id_to_oodid.items():
            ood_mask_copy[mask == k] = v

        seg_mask = Image.fromarray(seg_mask_copy.astype(np.uint8))
        ood_mask = Image.fromarray(ood_mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, seg_mask, ood_mask = xform(img, seg_mask, ood_mask=ood_mask, centroid=centroid)
                else:
                    img, seg_mask, ood_mask = xform(img, seg_mask, ood_mask=ood_mask)

        if self.transform is not None:
            img = self.transform(img)


        # img = normalize(img)

        if self.target_transform is not None:
            seg_mask = self.target_transform(seg_mask)

        seg_mask[seg_mask == 255] = 19
        return img, seg_mask.unsqueeze(0)

    def __len__(self):
        return len(self.imgs_uniform)
