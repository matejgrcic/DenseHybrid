from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import LostAndFoundDataset, LostAndFoundWithDistanceDataset
import torch

def _remap(x):
    l = torch.zeros_like(x)
    l[x == 2] = 1
    return l


def load_lost_and_found(dataroot, bs, train_transforms):
    remap_labels = tf.Lambda(lambda x: _remap((x*255.).long()))
    train_set = LostAndFoundDataset(dataroot, split='train',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=6)
    return train_loader

def load_lost_and_found_with_distance(dataroot, bs, train_transforms, val_transforms):
    remap_labels = tf.Lambda(lambda x: _remap((x*255.).long()))
    train_set = LostAndFoundWithDistanceDataset(dataroot, split='train',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    print(f"> Loaded {len(train_set)} train images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=6)
    test_set = LostAndFoundWithDistanceDataset(dataroot, split='test',
                                                image_transform=None if not val_transforms['image'] else tf.Compose(
                                                    val_transforms['image']),
                                                target_transform=None if not val_transforms['target'] else tf.Compose(
                                                    val_transforms['target'] + [remap_labels]),
                                                joint_transform=None if not val_transforms['joint'] else tf.Compose(
                                                    val_transforms['joint']))
    print(f"> Loaded {len(test_set)} test images.")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
    return train_loader, test_loader