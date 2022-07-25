from torchvision.datasets import Cityscapes as _Cityscapes
from torch.utils.data import Dataset

class Cityscapes(Dataset):
    def __init__(self, dataroot, split, image_transform, target_transform, joint_transform):
        super(Cityscapes, self).__init__()
        self._data = _Cityscapes(dataroot, split=split, mode='fine', target_type='semantic',
                       transform=image_transform, target_transform=target_transform)
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        img, lbl = self._data[index]
        return self.joint_transform((img, lbl)) if self.joint_transform else (img, lbl)
