from torchvision import transforms as tf
from torch.utils.data import DataLoader
from data.datasets import FSStaticDataset

def load_fs_static(dataroot, val_transforms):
    def process_label(x):
        x = (x*255.).long()
        x[x==255] = 2
        return x
    remap_labels = tf.Lambda(lambda x: process_label(x))
    val_set = FSStaticDataset(dataroot,
                                  image_transform=None if not val_transforms['image'] else tf.Compose(val_transforms['image']),
                                  target_transform=None if not val_transforms['target'] else tf.Compose(val_transforms['target'] + [remap_labels]),
                                  joint_transform=None if not val_transforms['joint'] else tf.Compose(val_transforms['joint']))
    print(f"> Loaded {len(val_set)} test images.")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    return val_loader
