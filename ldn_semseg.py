import os
import torch
import argparse
from utils import Logger
from data import get_dataset, RandomHorizontalFlip, JitterRandomCrop
import torchvision.transforms as tf
from models import LadderDenseNet
from experiments import SemsegExperiment
from PIL import Image

parser = argparse.ArgumentParser('Semseg training')
parser.add_argument('--dataroot',
                    help='dataroot',
                    type=str,
                    default='.')
parser.add_argument('--batch_size',
                    help='number of images in a mini-batch.',
                    type=int,
                    default=16)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=12)
parser.add_argument('--epochs',
                    help='maximum number of training epoches.',
                    type=int,
                    default=120)
parser.add_argument('--lr',
                    help='initial learning rate.',
                    type=float,
                    default=4e-4)
parser.add_argument('--lr_min',
                    help='min learning rate.',
                    type=float,
                    default=1e-7)
parser.add_argument('--momentum',
                    help='beta1 in Adam optimizer.',
                    type=float,
                    default=0.9)
parser.add_argument('--decay',
                    help='beta2 in Adam optimizer.',
                    type=float,
                    default=0.999)
parser.add_argument('--exp_name',
                    help='experiment name',
                    type=str,
                    required=True)
args = parser.parse_args()

class Args:
    def __init__(self):
        self.last_block_pooling = 0

def load_imagenet(segmentator):
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    state = load_state_dict_from_url('https://download.pytorch.org/models/densenet121-a639ec97.pth')
    # state = load_state_dict_from_url('https://download.pytorch.org/models/densenet169-b2777c0a.pth')

    ldn_state = {}
    for k, v in state.items():
        if 'transition' not in k:
            k = k.replace('norm.', 'norm')
            k = k.replace('conv.', 'conv')
        ldn_state[k] = v
    miss, unex = segmentator.backbone.load_state_dict(ldn_state, strict=False)
    print('Missing:', len(miss), 'Unexpected:', len(unex))
    return segmentator


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"./logs/{args.exp_name}"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!')
    os.makedirs(exp_dir, exist_ok=True)

    CROP_SIZE = 768

    logger = Logger(f"{exp_dir}/log.txt")
    logger.log(str(args))

    train_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': [
            JitterRandomCrop(size=CROP_SIZE, scale=(0.5, 2), ignore_id=args.num_classes, input_mean=(84, 88, 95)),# streethazards mean
            RandomHorizontalFlip()
        ]
    }

    val_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': None
    }
    loaders = get_dataset('street-hazards-full')(args.dataroot, args.batch_size, train_transforms, val_transforms)

    model = LadderDenseNet(args=Args(), num_classes=args.num_classes, checkpointing=True).to(device)
    model = load_imagenet(model)

    backbone_params = list(model.backbone.parameters())
    upsample_params = list(model.upsample.parameters()) \
                      + list(model.spp.parameters()) \
                      + list(model.logits.parameters())

    lr_backbone = args.lr / 4.
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': upsample_params}
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-7)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args)
    experiment.start()

if __name__ == '__main__':
    main(args)