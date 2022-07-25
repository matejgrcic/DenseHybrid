from abc import ABC
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from data import colorize_cityscapes_labels, denormalize_city_image
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F
# import seaborn as sns

class OODEvaluation(ABC):
    def __init__(self, model, loader, device, ignore_id, logger):
        self.model = model
        self.loader = loader
        self.device = device
        self.ignore_id = ignore_id
        self.logger = logger

    def calculate_auroc(self,conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k

    def calculate_ood_scores(self, scale=1.):
        total_conf = []
        total_gt = []
        with tqdm(total=len(self.loader.dataset)) as progress_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.loader):
                    img, lbl = batch
                    img = img.to(self.device)
                    lbl = lbl[:, 0]
                    lbl = lbl.to(self.device)
                    with torch.no_grad():
                        conf_probs = self.compute_ood_probs(img, lbl)
                    if scale != 1.:
                        conf_probs = F.interpolate(conf_probs.unsqueeze(1), scale_factor=scale, mode='bilinear')[:, 0]
                        lbl = F.interpolate(lbl.unsqueeze(1).float(), scale_factor=scale, mode='nearest')[:, 0].long()
                    label = lbl.view(-1)
                    conf_probs = conf_probs.view(-1)
                    gt = label[label != 2].cpu()
                    total_gt.append(gt)
                    conf = conf_probs.cpu()[label != 2]
                    total_conf.append(conf)
                    progress_bar.update(1)

        total_gt = torch.cat(total_gt, dim=0).numpy()
        total_conf = torch.cat(total_conf, dim=0).numpy()
        AP = average_precision_score(total_gt, total_conf)
        roc_auc, fpr, threshold = self.calculate_auroc(total_conf, total_gt)
        print(threshold)
        self.logger.log(f"> Average precision: {round(AP*100., 2)}%")
        self.logger.log(f"> FPR: {round(fpr*100., 2)}%")
        self.logger.log(f"> AUROC: {round(roc_auc*100., 2)}%")

    def calculate_ood_scores_per_image(self):
        total_AP = []
        total_fpr = []
        total_roc = []
        for step, batch in enumerate(self.loader):
            img, lbl = batch
            img = img.to(self.device)
            lbl = lbl[:, 0]
            lbl = lbl.to(self.device)
            with torch.no_grad():
                conf_probs = self.compute_ood_probs(img, lbl)

            label = lbl.view(-1)
            conf_probs = conf_probs.view(-1)
            gt = label[label != 2].cpu()
            conf = conf_probs.cpu()[label != 2]
            item_ap = average_precision_score(gt, conf)
            total_AP.append(item_ap)
            roc_auc, fpr, t = self.calculate_auroc(conf, gt)
            total_roc.append(roc_auc)
            total_fpr.append(fpr)
        ap = float(np.nanmean(total_AP)) * 100.
        fpr = float(np.nanmean(total_fpr)) * 100.
        auroc = float(np.nanmean(total_roc)) * 100.
        self.logger.log(f"> Average precision: {round(ap, 2)}%")
        self.logger.log(f"> FPR: {round(fpr, 2)}%")
        self.logger.log(f"> AUROC: {round(auroc, 2)}%")

    def plot_confidence(self, out_dir):
        for step, batch in enumerate(self.loader):
            img, lbl = batch
            lbl = lbl[:, 0]
            img = img.to(self.device)
            with torch.no_grad():
                # conf_probs = 1 - self.compute_ood_probs(img, lbl)
                # self._plot_conf_grid(conf_probs.squeeze(0).permute(1, 0).cpu().numpy(), step, out_dir)
                conf_probs = self.compute_ood_probs(img, lbl)
                self._plot_conf_grid(conf_probs.squeeze(0).cpu().numpy(), step, out_dir)

    def get_conf_img(self, conf):
        minv = conf.min()
        maxv = conf.max()
        conf = (conf - minv) / (maxv - minv)
        conf_broad = np.reshape(conf, [conf.shape[0], conf.shape[1], 1])

        # conf_save = plt.get_cmap('jet')(conf)
        # conf_save = plt.get_cmap('RdBu')(conf)
        conf_save = plt.get_cmap()(conf)
        conf_save = (conf_save * 255).astype(np.uint8)[:, :, :3]
        img = conf_save
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def _plot_conf_grid(self, probs, i, out_dir):
        # min_v = probs.min()
        # max_v = probs.max()
        # prob_plots = np.zeros_like(probs)
        # (H, W) = probs.shape
        # for k in range(H):
        #     for l in range(W):
        #         prob_plots[k, l] = probs[k, W - l - 1]
        # xx, yy = np.mgrid[0: H: 1, 0:W:1]
        # f, ax = plt.subplots(figsize=(8, 6))
        # contour = ax.contourf(xx, yy, prob_plots, 25, cmap="RdBu",
        #                       vmin=min_v, vmax=max_v)
        # ax_c = f.colorbar(contour)
        # ax_c.set_label("$P(y = 0)$")
        # ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, .25, .5, .75, .9, .95, 1])
        #
        # ax.set(aspect="equal",
        #        xlim=(0, H), ylim=(0, W),
        #        xlabel="$X_1$", ylabel="$X_2$")
        # plt.savefig(f"{out_dir}/confs_{i}.png")
        # plt.clf()
        # plt.close()
        out = self.get_conf_img(probs) / 255.
        save_image(out, f"{out_dir}/confs_{i}.png")

    def compute_ood_probs(self, img, lbl):
        pass

    def make_preds(self, img):
        return self.model(img)[0].max(1)[1]

    def plot_labels(self, out_dir):
        with torch.no_grad():
            for step, batch in enumerate(self.loader):
                img, lbl = batch
                H, W = img.shape[-2:]
                img = img.to(self.device)
                lbl = lbl[:, 0]
                lbl = lbl.to(self.device)
                with torch.no_grad():
                    pred = self.make_preds(img)
                    color = colorize_cityscapes_labels(pred.cpu())

                # save_image(torch.cat((color.cpu(), denormalize_city_image(img[0].cpu())), dim=-1), f"{out_dir}/pred_lbl_{step}.png")
                save_image(color.cpu(), f"{out_dir}/pred_lbl_{step}.png")

    def plot_hist(self, scale=1., title='a'):
        total_conf = []
        total_gt = []
        with tqdm(total=len(self.loader.dataset)) as progress_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.loader):
                    img, lbl = batch
                    img = img.to(self.device)
                    lbl = lbl[:, 0]
                    lbl = lbl.to(self.device)
                    with torch.no_grad():
                        conf_probs = self.compute_ood_probs(img, lbl)
                    if scale != 1.:
                        conf_probs = F.interpolate(conf_probs.unsqueeze(1), scale_factor=scale, mode='bilinear')[:, 0]
                        lbl = F.interpolate(lbl.unsqueeze(1).float(), scale_factor=scale, mode='nearest')[:, 0].long()
                    label = lbl.view(-1)
                    conf_probs = conf_probs.view(-1)
                    gt = label[label != 2].cpu()
                    total_gt.append(gt)
                    conf = conf_probs.cpu()[label != 2]
                    total_conf.append(conf)
                    progress_bar.update(1)

        total_gt = torch.cat(total_gt, dim=0).numpy()
        total_conf = torch.cat(total_conf, dim=0).numpy()
        inl = total_conf[total_gt==0]
        outl = total_conf[total_gt==1]
        np.random.shuffle(inl)
        inl = inl[:2*outl.shape[0]]
        print(inl.shape, outl.shape)
        sns.set_theme(style="darkgrid")
        plt.hist([inl, outl], bins=15, label=['Inliers', 'Outliers'])
        # plt.hist(outl, bins=15, label='Outliers')
        # plt.hist(inl, bins=15, label='Inliers')
        plt.title(title)
        plt.legend()
        plt.savefig(title+'.png')
        plt.clf()

