from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import LostFoundDataset

def load_lost_found(dataroot, val_transforms):
    remap_labels = tf.Lambda(lambda x: (x*255.).long())
    val_set = LostFoundDataset(dataroot,
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return val_loader