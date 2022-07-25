import os
import torch
import argparse
from utils import Logger
from data import get_dataset, JitterRandomCrop, RandomHorizontalFlip, AVAILABLE_DATASETS, get_negative_dataset
from data.joint_transforms.transforms import JointResize
import torchvision.transforms as tf
from models import DeepWV3PlusTH
from experiments import SemsegJointNegativesTHExperiment


parser = argparse.ArgumentParser('Semseg finetune')
parser.add_argument('--dataroot',
                    help='dataroot',
                    type=str,
                    default='.')
parser.add_argument('--dataset',
                    help='dataset',
                    type=str,
                    default='cityscapes')
parser.add_argument('--batch_size',
                    help='number of images in a mini-batch.',
                    type=int,
                    default=12)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=19)
parser.add_argument('--epochs',
                    help='maximum number of training epoches.',
                    type=int,
                    default=10)
parser.add_argument('--lr',
                    help='initial learning rate.',
                    type=float,
                    default=1e-6)
parser.add_argument('--lr_min',
                    help='min learning rate.',
                    type=float,
                    default=1e-6)
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
parser.add_argument('--resume',
                    help='Resume experiment',
                    action='store_true',
                    default=False)
parser.add_argument('--beta',
                    help='loss beta',
                    type=float,
                    default=0.03)
parser.add_argument('--neg_dataroot',
                    help='negative dataroot',
                    type=str,
                    default='.')
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

def load_flow_params(state):
    _state = dict()
    for k, v in state.items():
        key = '.'.join(k.split('.')[2:])
        _state[key] = v
    return _state



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"./logs/{args.dataset}/{args.exp_name}"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/imgs", exist_ok=True)

    CROP_SIZE = 512

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
                    JitterRandomCrop(size=CROP_SIZE, scale=(0.5, 2), ignore_id=args.num_classes, input_mean=(73, 83, 72)), # city mean
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

    neg_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [
            tf.ToTensor(),
        ],
        'joint': [
            JointResize(384),
            JitterRandomCrop(size=192, scale=(0.5, 2), ignore_id=args.num_classes, input_mean=(107, 117, 120))
        ],
    }
    loaders = get_dataset(args.dataset)(args.dataroot, args.batch_size, train_transforms, val_transforms)
    neg_loader = get_negative_dataset('ade')(args.neg_dataroot, args.batch_size, neg_transforms)

    model = DeepWV3PlusTH(num_classes=args.num_classes).to(device)
    model.load_pretrained_weights_cv0()

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr}
    ], betas=(0.9, 0.999), eps=1e-7)


    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    experiment = SemsegJointNegativesTHExperiment(
        model, optimizer, loaders, args.epochs, logger, device, f"{exp_dir}/checkpoint.pt", args, f"{exp_dir}/imgs", neg_loader
    )

    experiment.start()
    logger.close()

if __name__ == '__main__':
    main(args)