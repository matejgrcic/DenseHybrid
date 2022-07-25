from .base import OODEvaluation
import torch.nn.functional as F
import torch

class THKLOODEvaluation(OODEvaluation):
    def __init__(self, model, loader, device, ignore_id, logger):
        super(THKLOODEvaluation, self).__init__(model, loader, device, ignore_id, logger)


    def compute_ood_probs(self, img, lbl):
        logit, logit_ood = self.model(img, lbl.size()[1:3])
        out = torch.nn.functional.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)
        p2 = out[:, 1] # p(~din|x)
        probs = (- p1) + p2.log()
        conf_probs = probs
        return conf_probs

