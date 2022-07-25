import os
import torch
import argparse
from utils import Logger
from data import get_eval_dataset, AVAILABLE_EVAL_DATASETS
import torchvision.transforms as tf
from models import LadderDenseNetTH, DeepWV3PlusTH
from evaluations import THKLOODEvaluation

parser = argparse.ArgumentParser('Dense anomaly detection eval')
parser.add_argument('--dataroot',
                    help='dataroot',
                    type=str,
                    default='.')
parser.add_argument('--dataset',
                    help='dataset',
                    type=str,
                    choices=AVAILABLE_EVAL_DATASETS)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=19)
parser.add_argument('--folder',
                    help='output folder',
                    type=str,
                    required=True)
parser.add_argument('--params',
                    help='weights file',
                    type=str,
                    required=True)
args = parser.parse_args()

class Args:
    def __init__(self):
        self.last_block_pooling = 0

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir = f"{args.folder}/eval"
    img_dir = f"{args.folder}/eval/imgs"
    if os.path.exists(exp_dir):
        raise Exception('Directory exists!', exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    logger = Logger(f"{exp_dir}/log_eval.txt")
    logger.log(str(args))


    val_transforms = {
        'image': [
            tf.ToTensor(),
        ],
        'target': [tf.ToTensor()],
        'joint': None
    }
    loaders = get_eval_dataset(args.dataset)(args.dataroot, val_transforms)

    if args.dataset == 'street-hazards':
        model = LadderDenseNetTH(args=Args(), num_classes=12, checkpointing=True).to(device)
    else:
        model = DeepWV3PlusTH(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.params), strict=True)
    model.eval()
    logger.log("> Loaded model.")

    logger.log("== DenseHybrid anomaly detection ==")
    experiment = THKLOODEvaluation(model, loaders, device, ignore_id=2, logger=logger)
    if args.dataset == 'street-hazards':
        experiment.calculate_ood_scores_per_image()
    else:
        experiment.calculate_ood_scores(1)

if __name__ == '__main__':
    main(args)