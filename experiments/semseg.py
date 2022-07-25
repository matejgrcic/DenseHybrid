import torch
import torch.nn.functional as F
import math
from .base import Experiment
from utils import IoU
from tqdm import tqdm

class SemsegExperiment(Experiment):

    def train(self):
        self.logger.log(f"Epoch {self.current_epoch}")
        self.model.train()
        running_loss = 0.
        metrics = IoU(self.args.num_classes + 1, ignore_index=self.args.num_classes)
        with tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            for batch_idx, data in enumerate(self.train_loader, 1):
                self.optimizer.zero_grad()

                x, label = data
                x = x.to(self.device)
                label = label[:, 0].to(self.device)

                logits = self.model(x, label.size()[1:3])
                cls_out = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(cls_out, label, ignore_index=self.args.num_classes)
                loss.backward()
                self.optimizer.step()

                pred = cls_out.max(1)[1]
                metrics.add(pred, label)

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(x.size(0))

        mean_loss = running_loss / batch_idx
        self.logger.log(f"> Average train loss: {round(mean_loss, 2)}")
        _, miou = metrics.value()
        self.logger.log(f"===> Train mIoU: {round(miou*100., 2)}%")
        torch.cuda.empty_cache()
        if hasattr(self.train_loader.dataset, 'build_epoch'):
            self.train_loader.dataset.build_epoch()
            self.logger.log("> Builded new epoch!")

    def eval(self):
        running_loss = 0.
        self.model.eval()
        metrics = IoU(self.args.num_classes + 1, ignore_index=self.args.num_classes)
        correct = 0
        total = 0
        with tqdm(total=len(self.val_loader.dataset)) as progress_bar:
            with torch.no_grad():
                for batch_idx, data in enumerate(self.val_loader, 1):
                    x, y = data
                    x = x.to(self.device)
                    y = y[:, 0]
                    y = y.to(self.device)

                    cls_out = self.model(x, y.size()[1:3])
                    cls_out = F.log_softmax(cls_out, dim=1)
                    loss = F.nll_loss(cls_out, y, ignore_index=self.args.num_classes)
                    running_loss += loss.item()

                    pred = cls_out.max(1)[1]
                    metrics.add(pred, y)
                    correct += pred.eq(y).cpu().sum().item()
                    total += y.size(0) * y.size(1) * y.size(2)
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(x.size(0))

                mean_loss = running_loss / batch_idx
                self.logger.log(f"> Average validation loss: {round(mean_loss,2)}")
                self.logger.log(f"> Average validation accuracy: {round(correct * 100. / total, 2)}%")
                iou, miou = metrics.value()
                self.logger.log(f"> Validation mIoU: {round(miou * 100., 2)}%")
        torch.cuda.empty_cache()
        return miou

    def process_lr(self):
        lr = self.args.lr_min \
             + (self.args.lr - self.args.lr_min) * (1 + math.cos(self.current_epoch / self.epochs * math.pi)) / 2
        self.optimizer.param_groups[0]['lr'] = lr / 4
        self.optimizer.param_groups[1]['lr'] = lr
