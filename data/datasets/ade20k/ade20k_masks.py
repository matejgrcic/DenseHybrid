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


class ADE20KMasks(data.Dataset):

    def __init__(self, root, split, target_transform):

        self.root = root

        self.images_base = os.path.join(self.root, 'images', split)
        self.annotations_base = os.path.join(self.root, 'annotations', split)
        self.filenames = recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.target_transform = target_transform

        print("> Found %d masks..." % (len(self.filenames)))


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):

        filename = self.filenames[index].replace('.jpg', '.png')
        lbl_path = os.path.join(self.annotations_base, filename)

        label = Image.open(lbl_path)
        label = self.target_transform(label)
        label = (label * 255.).long()
        return label
