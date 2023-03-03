import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2 as cv
from road_anomaly_benchmark.evaluation import Evaluation
import torchvision.transforms as tf
import argparse
from models import LadderDenseNetTH

IMG_SIZE = 1024
to_tensor = tf.ToTensor()
resize = tf.Resize(IMG_SIZE)

class Args:
    def __init__(self):
        self.last_block_pooling = 0

datasets = ['ObstacleTrack-validation', 'ObstacleTrack-all', 'AnomalyTrack-validation', 'AnomalyTrack-all','LostAndFound-testNoKnown', 'LostAndFound-test']
parser = argparse.ArgumentParser('Evaluations')
parser.add_argument('--file',
                    help='cp file',
                    type=str,
                    required=True)
parser.add_argument('--dataset',
                    help='dataset',
                    type=str,
                    choices=datasets,
                    required=True)
parser.add_argument('--name',
                    help='exp_name',
                    type=str,
                    required=True)
parser.add_argument('--num_classes',
                    help='num classes of segmentator.',
                    type=int,
                    default=19)
parser.add_argument('--use_mask',
                    help='Resume experiment',
                    action='store_true',
                    default=False)


def method_densehybrid(image, model, args):
    image = to_tensor(image)
    H, W = image.shape[-2:]
    # image = resize(image)
    with torch.no_grad():
        logit, logit_ood = model(image.unsqueeze(0).cuda(), (H, W))
        out = torch.nn.functional.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1]
        probs = (- p1) + p2.log()
    conf_probs = probs

    return conf_probs.squeeze().cpu().numpy()

def main(args):
    model = LadderDenseNetTH(args=Args(), num_classes=args.num_classes, checkpointing=True).cuda()
    model.load_state_dict(torch.load(args.file), strict=True)
    model.eval()

    ev = Evaluation(
        method_name = args.name,
        dataset_name=args.dataset
    )

    for frame in tqdm(ev.get_frames()):
        # run method here
        result = method_densehybrid(frame.image, model, args)
        # provide the output for saving
        ev.save_output(frame, result)

    # wait for the background threads which are saving
    ev.wait_to_finish_saving()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
