from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class StreetHazardsTestBase(Dataset):

    def __init__(self, dataroot, split, city, image_transform, target_transform):
        super(StreetHazardsTestBase, self).__init__()

        assert split in ['test']
        assert city in ['t5', 't6', 'both']
        self.split = split
        self.city = city
        self.img_paths, self.lbl_paths = self.load_paths(dataroot, split)

        self.image_transform = image_transform
        self.target_transform = target_transform

        if len(self.img_paths) == 0:
            raise Exception("> No files for split=[%s] found in %s" % (split, dataroot))

        print("> Found %d %s images..." % (len(self.img_paths), split))

    def load_paths(self, dataroot, split):
        images = []
        labels = []
        root = os.path.join(dataroot, 'test' if split == 'test' else 'train', 'images', split)
        for looproot, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith('png'):
                    continue
                if self.city != 'both' and self.city in looproot:
                    continue
                images.append(os.path.join(looproot, filename))
                labels.append(os.path.join(looproot, filename).replace('images', 'annotations'))
        return images, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        lbl_path = self.lbl_paths[index]

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        image = Image.open(img_path)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        label = Image.open(lbl_path)

        image = self.image_transform(image)
        label = self.target_transform(label)
        label = (label * 255.).long()

        ood_label = torch.zeros_like(label)
        ood_label[label == 14] = 1
        ood_label[label != 14] = 0
        ood_label[label == 4] = 2

        image = image[:3]
        label = label - 1
        label[label == 3] = 13
        label[label >= 3] = label[label >= 3] - 1
        # label: 12 class ([0, 11]), ignore class [12]
        # ood_label: [0]-OK, [1]-Anomaly [2]-Ignore
        return image, label, ood_label

class StreetHazardsTestOOD(StreetHazardsTestBase):
    def __init__(self, dataroot, split, city, image_transform, target_transform, joint_transform):
        super(StreetHazardsTestOOD, self).__init__(dataroot, split, city, image_transform, target_transform)

        self.joint_transform = joint_transform

    def __getitem__(self, index):
        image, _, ood_label = super(StreetHazardsTestOOD, self).__getitem__(index)
        return self.joint_transform((image, ood_label)) if self.joint_transform else (image, ood_label)


class StreetHazardsTestSeg(StreetHazardsTestBase):
    def __init__(self, dataroot, split, city, image_transform, target_transform, joint_transform):
        super(StreetHazardsTestSeg, self).__init__(dataroot, split, city, image_transform, target_transform)

        self.joint_transform = joint_transform

    def __getitem__(self, index):
        image, label, _ = super(StreetHazardsTestSeg, self).__getitem__(index)
        return self.joint_transform((image, label)) if self.joint_transform else (image, label)

class StreetHazardsOSR(StreetHazardsTestBase):
    def __init__(self, dataroot, split, city, image_transform, target_transform, joint_transform):
        super(StreetHazardsOSR, self).__init__(dataroot, split, city, image_transform, target_transform)

        self.joint_transform = joint_transform

    def __getitem__(self, index):
        image, label, ood_label = super(StreetHazardsOSR, self).__getitem__(index)
        label[ood_label == 1] = 12
        label[ood_label == 2] = 13
        # OSR labels: 12 inlier classes [0, 11], anomaly class [12], ignore [13]
        return self.joint_transform((image, label)) if self.joint_transform else (image, label)
