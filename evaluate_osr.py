import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
from data import colorize_streethazards_labels, load_street_hazards_osr
from models import LadderDenseNetTH
import argparse
from utils import Logger, IoU, OpenIoU
import torchvision.transforms as tf
import torch.nn.functional as F
import torch.nn as nn
class Args:
    def __init__(self):
        self.last_block_pooling = 0


class OpenModel(nn.Module):
    def __init__(self, model, threshold=None):
        super(OpenModel, self).__init__()
        self.model = model
        self.register_buffer('threshold', threshold)

    def set_threshold(self, t):
        self.register_buffer('threshold', t)

    def ood_score(self, img, shape=None):
        logit, logit_ood = self.model(img, shape)
        out = F.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1]  # p(~din|x)
        conf_probs = (- p1) + p2.log()
        return conf_probs

    def forward(self, img, shape=None):
        assert self.threshold != None
        logit, logit_ood = self.model(img, shape)
        out = F.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1]  # p(~din|x)
        conf_probs = (- p1) + p2.log()  # - ln hat_p(x, din) + ln p(~din|x)
        classes = logit.max(1)[1]
        classes[conf_probs > self.threshold] = logit.size(1)
        return classes

class OODCalibration:
    def __init__(self, model, loader, device, ignore_id, logger):
        self.model = model
        self.loader = loader
        self.device = device
        self.ignore_id = ignore_id
        self.logger = logger

    def calculate_stats(self,conf, gt, rate=0.95):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        treshold = 0
        for i, j, k in zip(tpr, fpr, threshold):
            if i > rate:
                fpr_best = j
                treshold = k
                break
        return roc_auc, fpr_best, treshold

    def calculate_ood_scores(self, desired_tpr=0.95, scale=1.):
        total_conf = []
        total_gt = []
        with torch.no_grad():
            for step, batch in enumerate(self.loader):
                img, lbl = batch
                img = img.to(self.device)
                lbl = lbl[:, 0]
                lbl = lbl.to(self.device)
                ood_lbl = torch.zeros_like(lbl)
                ood_lbl[lbl == 12] = 1
                ood_lbl[lbl == 13] = 2
                lbl = ood_lbl
                with torch.no_grad():
                    conf_probs = self.model.ood_score(img, lbl.shape[1:])
                if scale != 1.:
                    conf_probs = F.interpolate(conf_probs.unsqueeze(1), scale_factor=scale, mode='bilinear')[:, 0]
                    lbl = F.interpolate(lbl.unsqueeze(1).float(), scale_factor=scale, mode='nearest')[:, 0].long()

                label = lbl.view(-1)
                conf_probs = conf_probs.view(-1)
                gt = label[label != 2].cpu()
                total_gt.append(gt)
                conf = conf_probs.cpu()[label != 2]
                total_conf.append(conf)

        total_gt = torch.cat(total_gt, dim=0).numpy()
        total_conf = torch.cat(total_conf, dim=0).numpy()
        AP = average_precision_score(total_gt, total_conf)
        roc_auc, fpr, treshold = self.calculate_stats(total_conf, total_gt, rate=desired_tpr)
        # self.logger.log(f"> Average precision: {round(AP*100., 2)}%")
        # self.logger.log(f"> FPR: {round(fpr*100., 2)}%")
        # self.logger.log(f"> AUROC: {round(roc_auc*100., 2)}%")
        # self.logger.log(f"> Treshold: {round(treshold, 2)}")
        return treshold

def evaluate_open_dataset(loader, model):
    metrics = OpenIoU(14, ignore_index=13)
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()[:, 0]
        with torch.no_grad():
            preds = model(x, y.shape[1:])
        metrics.add(preds, y)

    iou = metrics.iou_value()
    miou = np.nanmean(iou[:-2])
    print(f"OPEN SET: mIoU over 12 classes {miou * 100.}")
    print(f"OPEN SET: anomaly class IoU {iou[-2]*100.}")
    return miou

def evaluate_closed_dataset(loader, model):
    metrics = IoU(13, ignore_index=12)
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()[:, 0]
        with torch.no_grad():
            logits, _ = model.forward(x, y.shape[1:])
        preds = logits.max(1)[1]
        y[y >= 12] = 12
        metrics.add(preds, y)
    iou, miou = metrics.value()
    print(f"CLOSED SET: mIoU over 12 classes {miou * 100.}")

def compute_osr_perf(loader, loader_anom, model, desired_tpr=0.9, scale=0.5):
    model = OpenModel(model).cuda()
    model.eval()

    calibrator = OODCalibration(model, loader_anom, 'cuda', ignore_id=2, logger=logger)
    treshold = calibrator.calculate_ood_scores(desired_tpr=desired_tpr, scale=scale)
    model.set_threshold(torch.tensor(treshold))
    miou = evaluate_open_dataset(loader, model.eval())
    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate dense OSR')
    parser.add_argument('--dataroot',
                        help='dataroot',
                        type=str,
                        default='.')
    parser.add_argument('--num_classes',
                        help='num classes of segmentator.',
                        type=int,
                        default=12)
    parser.add_argument('--tpr',
                        help='num classes of segmentator.',
                        type=float,
                        default=0.95)
    parser.add_argument('--model',
                        help='cp file',
                        type=str,
                        required=True)
    args = parser.parse_args()

    model = LadderDenseNetTH(args=Args(), num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(args.model), strict=True)
    model.eval()


    exp_dir = '/'.join(args.model.split('/')[:-1])
    logger = Logger(f"{exp_dir}/log_eval.txt")
    logger.log(str(args))

    val_transforms = {
        'image': [tf.ToTensor()],
        'target': [tf.ToTensor()],
        'joint': None
    }
    loader_t5 = load_street_hazards_osr(args.dataroot, 't5', val_transforms)
    loader_t6 = load_street_hazards_osr(args.dataroot, 't6', val_transforms)
    loader_all = load_street_hazards_osr(args.dataroot, 'both', val_transforms)
    print('>>> Performance on closed set')
    evaluate_closed_dataset(loader_all, model)

    print('>>> Performance on t5')
    miou_t5 = compute_osr_perf(loader_t5, loader_t6, model, desired_tpr=args.tpr)
    print('>>> Performance on t6')
    miou_t6 = compute_osr_perf(loader_t6, loader_t5, model, desired_tpr=args.tpr)

    print('>>> Final performance')
    t5_len = len(loader_t5.dataset)
    t6_len = len(loader_t6.dataset)
    miou = ((t5_len * miou_t5) + (t6_len * miou_t6)) / (t5_len + t6_len) # weighted average
    print(f"OPEN SET: mIoU over 12 classes {miou * 100.}")














