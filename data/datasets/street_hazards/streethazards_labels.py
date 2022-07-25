import numpy as np
import torch

_color_info = [[[0, 0, 0], ['background']], # 0
                  [[70, 70, 70], ['building']], # 1
                  [[190, 153, 153], ['fence']], # 2
                 [[220, 20, 60], ['person']], # 3
                  [[153, 153, 153], ['pole']], # 4
                 [[255, 255, 255], ['streetlines']], # 5
                 [[128, 64, 128], ['road']], # 6
                  [[244, 35, 232], ['sidewalk']], # 7
                 [[107, 142, 35], ['vegetation']], # 8
                 [[0, 0, 142], ['car']], # 9
                 [[102, 102, 156], ['wall']], # 10
                  [[220, 220, 0], ['traffic sign']], # 11
                    [[60, 250, 240], ['anomaly']],
               [[ 0, 0, 0], ['ignore']]]


class ColorizeLabels:
    def __init__(self):
        color_info = _color_info
        self.color_info = np.array(color_info)

    def _trans(self, lab):
        R, G, B = [np.zeros_like(lab) for _ in range(3)]
        for l in np.unique(lab):
            mask = lab == l
            R[mask] = self.color_info[l][0][0]
            G[mask] = self.color_info[l][0][1]
            B[mask] = self.color_info[l][0][2]
        return torch.LongTensor(np.stack((R, G, B), axis=-1).astype(np.uint8)).squeeze().permute(2, 0, 1).float() / 255.

    def __call__(self, example):
        return self._trans(example)

colorize_labels = ColorizeLabels()



