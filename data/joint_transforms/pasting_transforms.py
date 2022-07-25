import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torchvision.transforms.functional as F
from enum import Enum
import torch
from torch import Tensor
import numpy as np
import cv2
from PIL import Image


class OutlierInjection(torch.nn.Module):

    def __init__(self, data_loader, per_image_outliers, ood_id):
        super(OutlierInjection, self).__init__()
        self.data_loader = data_loader
        self.outlier_iter = iter(data_loader)
        self.per_image_outliers = per_image_outliers
        self.ood_id = ood_id

    def _get_outlier(self):
        try:
            outlier = next(self.outlier_iter)
        except:
            self.outlier_iter = iter(self.data_loader)
            outlier = next(self.outlier_iter)
        return outlier[0].squeeze()

    def _paste_outlier(self, outlier, img, lbl):
        w, h = F.get_image_size(img)
        o_w, o_h = F.get_image_size(outlier)

        i = torch.randint(0, h - o_h, size=(1,)).item()
        j = torch.randint(0, w - o_w, size=(1,)).item()

        img[:, i:i + o_h, j:j+o_w] = outlier
        lbl[:, i:i + o_h, j:j + o_w] = self.ood_id
        return img, lbl



    def forward(self, data):
        img, lbl = data
        for _ in range(self.per_image_outliers):
            img, lbl = self._paste_outlier(self._get_outlier(), img, lbl)
        return img, lbl

class OutlierInjectionWithoutInlier(torch.nn.Module):

    def __init__(self, data_loader, per_image_outliers, ood_id, ignore_id, p=0.5):
        super(OutlierInjectionWithoutInlier, self).__init__()
        self.data_loader = data_loader
        self.outlier_iter = iter(data_loader)
        self.per_image_outliers = per_image_outliers
        self.ood_id = ood_id
        self.ignore_id = ignore_id
        self.p = p

    def _get_outlier(self):
        try:
            outlier = next(self.outlier_iter)
        except:
            self.outlier_iter = iter(self.data_loader)
            outlier = next(self.outlier_iter)
        return outlier[0].squeeze()

    def _paste_outlier(self, outlier, img, lbl):
        w, h = F.get_image_size(img)
        o_w, o_h = F.get_image_size(outlier)

        i = torch.randint(0, h - o_h, size=(1,)).item()
        j = torch.randint(0, w - o_w, size=(1,)).item()

        if torch.rand(1) < self.p:
            img = torch.zeros_like(img)
            lbl = torch.ones_like(lbl) * self.ignore_id

        img[:, i:i + o_h, j:j+o_w] = outlier
        lbl[:, i:i + o_h, j:j + o_w] = self.ood_id
        return img, lbl



    def forward(self, data):
        img, lbl = data
        for _ in range(self.per_image_outliers):
            img, lbl = self._paste_outlier(self._get_outlier(), img, lbl)
        return img, lbl

class OutlierInjectionWithoutInlierAndWithMask(torch.nn.Module):

    def __init__(self, data_loader, per_image_outliers, ood_id, ignore_id, p=0.5):
        super(OutlierInjectionWithoutInlierAndWithMask, self).__init__()
        self.data_loader = data_loader
        self.outlier_iter = iter(data_loader)
        self.per_image_outliers = per_image_outliers
        self.ood_id = ood_id
        self.ignore_id = ignore_id
        self.p = p

    def _get_outlier(self):
        try:
            outlier = next(self.outlier_iter)
        except:
            self.outlier_iter = iter(self.data_loader)
            outlier = next(self.outlier_iter)
        return outlier[0][0], outlier[1][0]

    def _paste_outlier_instance(self, patch, patch_lbl, outlier, mask, idx):
        _, H, W = patch_lbl.shape
        mask = mask[:, :H, :W]
        outlier = outlier[:, :H, :W]
        patch_lbl[mask == idx] = self.ood_id

        mask = mask.repeat(3, 1, 1)
        patch[mask == idx] = outlier[mask == idx]
        return patch, patch_lbl

    def _get_size_for_class(self, locations, mask, outlier):
        h_min = locations[0].min()
        h_max = locations[0].max()
        w_min = locations[1].min()
        w_max = locations[1].max()

        height, width = h_max - h_min + 1, w_max - w_min + 1
        o = outlier[:, h_min:h_min + height, w_min:w_min + width]
        m = mask[:, h_min:h_min + height, w_min:w_min + width]
        return o, m, (height, width)

    def _scale_outlier(self, negative, mask, coef, h, w):
        neg = F.resize(negative, (int(h*coef)+1, int(w*coef)+1), Image.BILINEAR)
        mask = F.resize(mask, (int(h*coef)+1, int(w*coef)+1), Image.NEAREST)
        return neg, mask, (int(h*coef)+1, int(w*coef)+1)



    def _paste_outlier(self, negative, img, lbl):
        w, h = F.get_image_size(img)
        outlier, mask = negative
        o_w, o_h = F.get_image_size(outlier)

        # if w <= o_w or h <= o_h:
        #     outlier = F.resize(outlier, int(0.5 * min(h, w)), Image.BILINEAR)
        #     mask = F.resize(mask, int(0.5 * min(h, w)), Image.NEAREST)
        #     o_w, o_h = F._get_image_size(outlier)

        indices = list(set(mask.view(-1).cpu().tolist()))
        i1 = np.random.choice(indices, 1)[0]

        locations = np.where(mask[0].numpy() == i1)
        outlier, mask, (o_h, o_w) = self._get_size_for_class(locations, mask, outlier)

        scale = random.randint(1, 15) / 100.0
        coef = scale * math.sqrt((h * w) / (o_h * o_w))
        outlier, mask, (o_h, o_w) = self._scale_outlier(outlier, mask, coef, o_h, o_w)

        if h - o_h > 0:
            i = torch.randint(0, h - o_h, size=(1,)).item()
        else:
            i = 0

        if w - o_w > 0:
            j = torch.randint(0, w - o_w, size=(1,)).item()
        else:
            j = 0

        patch = img[:, i:i + o_h, j:j+o_w]
        patch_lbl = lbl[:, i:i + o_h, j:j + o_w]
        patch, patch_lbl = self._paste_outlier_instance(patch, patch_lbl, outlier, mask, i1)
        img[:, i:i + o_h, j:j+o_w] = patch
        lbl[:, i:i + o_h, j:j + o_w] = patch_lbl
        return img, lbl

    def forward(self, data):
        img, lbl = data
        outliers_num = self.per_image_outliers
        if torch.rand(1) < self.p:
            img = torch.zeros_like(img)
            lbl = torch.ones_like(lbl) * self.ignore_id
            outliers_num *= 2
        for _ in range(outliers_num):
            img, lbl = self._paste_outlier(self._get_outlier(), img, lbl)
        return img, lbl
