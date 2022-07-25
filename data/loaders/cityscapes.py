from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import Cityscapes, create_id_to_train_id_mapper, load_city_uniform


def load_cityscapes(dataroot, bs, train_transforms, val_transforms):
    label_mapper = create_id_to_train_id_mapper()
    remap_labels = tf.Lambda(lambda x: label_mapper[(x*255.).long()])
    train_set = Cityscapes(dataroot, split='train',
                                  image_transform=None if not train_transforms['image'] else tf.Compose(train_transforms['image']),
                                  target_transform=None if not train_transforms['target'] else tf.Compose(train_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not train_transforms['joint'] else tf.Compose(train_transforms['joint']))
    print(f"> Loaded {len(train_set)} train images.")
    val_set = Cityscapes(dataroot, split='val',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, val_loader

def load_cityscapes_uniform_loader(dataroot, bs, train_transforms, val_transforms):
    label_mapper = create_id_to_train_id_mapper()
    remap_labels = tf.Lambda(lambda x: label_mapper[(x * 255.).long()])
    train_set = load_city_uniform(dataroot)
    print(f"> Loaded {len(train_set)} train images.")
    val_set = Cityscapes(dataroot, split='val',
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} val images.")
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, val_loader