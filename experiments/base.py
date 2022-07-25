from abc import ABC
import torch
import os
import math
class Experiment(ABC):

    def __init__(self, model, optimizer, loaders, epochs, logger, device, cp_file, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.logger = logger
        self.device = device
        self.current_epoch = 0
        self.epochs = epochs
        self.cp_file = cp_file
        self.args = args
        self.best = - math.inf

    def resume(self):
        self.load_checkpoint()
        self.start()

    def start(self):
        while self.current_epoch < self.epochs:
            self.train()
            metric = self.eval()
            self.current_epoch += 1
            self.process_lr()
            self.store_checkpoint()
            if metric > self.best:
                self.store_best()
                self.best = metric
            # if self.current_epoch >= self.epochs - 10 or self.current_epoch % 10 == 0:
            if self.current_epoch >= 5:
                self.store_model()

    def train(self):
        pass

    def eval(self):
        pass

    def process_lr(self):
        pass

    def store_checkpoint(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epochs': self.epochs,
            'current_epoch': self.current_epoch
        }, self.cp_file)

    def load_checkpoint(self):
        cp = torch.load(self.cp_file)
        self.model.load_state_dict(cp['model'])
        self.optimizer.load_state_dict(cp['optimizer'])
        self.epochs = cp.epochs
        self.current_epoch = cp.current_epoch

    def store_best(self):
        path = self.cp_file.split('/')
        dest = '/'.join(path[:-1] + ['model_best.pth'])
        torch.save(self.model.state_dict(), dest)
        self.logger.log(f"++ Epoch {self.current_epoch-1}: Saved best model so far.")

    def store_model(self):
        path = self.cp_file.split('/')
        dest = '/'.join(path[:-1] + [f"model_{self.current_epoch}.pth"])
        torch.save(self.model.state_dict(), dest)
        self.logger.log(f"-- Epoch {self.current_epoch - 1}: Saved model.")
