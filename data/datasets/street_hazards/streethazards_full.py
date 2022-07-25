from torch.utils.data import Dataset
import os
from PIL import Image

# mean = 84, 88, 95
class StreetHazardsFull(Dataset):

    def __init__(self, dataroot, split, image_transform, target_transform, joint_transform):
        super(StreetHazardsFull, self).__init__()

        assert split in ['training', 'validation']

        self.split = split
        self.img_paths, self.lbl_paths = self.load_paths(dataroot, split)

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

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
        image = image[:3]
        label = label - 1
        label[label == 3] = 13
        label[label >= 3] = label[label >= 3] - 1
        return self.joint_transform((image, label)) if self.joint_transform else (image, label)
