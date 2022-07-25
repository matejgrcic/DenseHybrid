from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import ADE20K

def load_ade_negatives(dataroot, bs, transforms):
    remap_labels = tf.Lambda(lambda x: (x*255.).long())
    negative_set = ADE20K(dataroot, split='training',
                                  image_transform=None if not transforms['image'] else tf.Compose(transforms['image']),
                                  target_transform=None if not transforms['target'] else tf.Compose(transforms['target'] + [remap_labels]),
                                  joint_transform=None if not transforms['joint'] else tf.Compose(transforms['joint']))
    print(f"> Loaded {len(negative_set)} negative images.")
    loader = DataLoader(negative_set, batch_size=bs, shuffle=True, drop_last=True)
    return loader