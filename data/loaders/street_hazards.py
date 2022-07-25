from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import StreetHazardsTestOOD, StreetHazardsFull, StreetHazardsOSR

def load_street_hazards_full(dataroot, bs, train_transforms, val_transforms, num_workers=3):
    train_set = StreetHazardsFull(dataroot, split='training',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target']),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    print(f"> Loaded {len(train_set)} train images.")
    val_set = StreetHazardsFull(dataroot, split='validation',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers//2)
    return train_loader, val_loader

def load_street_hazards_ood(dataroot, val_transforms):
    val_set = StreetHazardsTestOOD(dataroot, split='test', city='both',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return val_loader

def load_street_hazards_osr(dataroot, city, val_transforms, split='test'):
    ood_set = StreetHazardsOSR(dataroot, split=split, city=city,
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target']),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(ood_set)} test images.")
    loader = DataLoader(ood_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
    return loader