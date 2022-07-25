import os
from PIL import Image
from torch.utils import data

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [filename
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if
            (filename.endswith(suffix) and os.path.isfile(os.path.join(looproot, filename)))]


class ADE20K(data.Dataset):

    def __init__(self, root, split, image_transform, target_transform, joint_transform):

        self.root = root

        self.images_base = os.path.join(self.root, 'images', split)
        self.annotations_base = os.path.join(self.root, 'annotations', split)
        self.filenames = recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        print("> Found %d images..." % (len(self.filenames)))


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):

        img_path = os.path.join(self.images_base, self.filenames[index])
        filename = self.filenames[index].replace('.jpg', '.png')
        lbl_path = os.path.join(self.annotations_base, filename)

        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)

        label = Image.open(lbl_path)
        label = self.target_transform(label)

        return self.joint_transform((image, label)) if self.joint_transform else (image, label)
