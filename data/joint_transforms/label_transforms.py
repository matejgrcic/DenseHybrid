import torch
import numpy as np
import cv2

class LabelToBoundaryClone(torch.nn.Module):
    def __init__(self, num_classes, ignore_id, bins=(4, 16, 64, 128), alphas=(8., 6., 4., 2., 1.)):
        super(LabelToBoundaryClone, self).__init__()
        self.num_classes = num_classes
        self.bins = bins
        self.alphas = alphas
        self.ignore_id = ignore_id

    def forward(self, data):
        img, lbl = data
        labels = lbl.numpy()[0]
        present_classes = np.unique(labels)
        distances = np.zeros([self.num_classes] + list(labels.shape), dtype=np.float32) - 1.
        for i in range(self.num_classes):
            if i not in present_classes:
                continue
            class_mask = labels == i
            distances[i][class_mask] = cv2.distanceTransform(np.uint8(class_mask), cv2.DIST_L2, maskSize=5)[class_mask]
        ignore_mask = labels == self.ignore_id
        distances[distances < 0] = 0
        distances = distances.sum(axis=0)
        label_distance_bins = np.digitize(distances, self.bins)
        label_distance_alphas = np.zeros(label_distance_bins.shape, dtype=np.float32)
        for idx, alpha in enumerate(self.alphas):
            label_distance_alphas[label_distance_bins == idx] = alpha
        label_distance_alphas[ignore_mask] = 0
        return img, lbl, label_distance_alphas