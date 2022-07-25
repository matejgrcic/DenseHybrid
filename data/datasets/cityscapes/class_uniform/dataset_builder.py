"""
Dataset setup and loaders
"""

import torchvision.transforms as standard_transforms

from .joint_transforms import RandomSizeAndCrop, RandomHorizontallyFlip, Resize, Compose
from .transforms import RandomGaussianBlur, MaskToTensor
from torch.utils.data import DataLoader, ConcatDataset
import torch

from .cityscapes import CityScapesUniform


num_classes = 19
ignore_label = 255


def get_train_joint_transform():
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """
    # Geometric image transformations
    train_joint_transform_list = []
    train_joint_transform_list += [
        RandomSizeAndCrop(768,
                           crop_nopad=False,
                           pre_size=None,
                           scale_min=0.5,
                           scale_max=2,
                           ignore_index=ignore_label),
        Resize(768),
        RandomHorizontallyFlip()]

    # if args.rrotate > 0:
    #     train_joint_transform_list += [joint_transforms.RandomRotate(
    #         degree=args.rrotate,
    #         ignore_index=dataset.ignore_label)]

    train_joint_transform = Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform


def get_input_transforms():
    """
    Get input transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_input_transform, val_input_transform
    """
    # Image appearance transformations
    train_input_transform = []
    val_input_transform = []
    # if args.color_aug > 0.0:
    train_input_transform += [standard_transforms.RandomApply([
        standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)]

    # if args.bblur:
    #     train_input_transform += [extended_transforms.RandomBilateralBlur()]
    # elif args.gblur:
    train_input_transform += [RandomGaussianBlur()]

    train_input_transform += [
                              standard_transforms.ToTensor()
    ]
    val_input_transform += [
                            standard_transforms.ToTensor()
    ]
    train_input_transform = standard_transforms.Compose(train_input_transform)
    val_input_transform = standard_transforms.Compose(val_input_transform)

    return train_input_transform, val_input_transform


def get_target_transforms():
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """
    target_transform = MaskToTensor()
    # if args.jointwtborder:
    #     target_train_transform = extended_transforms.RelaxedBoundaryLossToTensor(
    #             dataset.ignore_label, dataset.num_classes)
    # else:
    target_train_transform = MaskToTensor()

    target_aux_train_transform = MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform


def load_city_uniform(dataroot):
    """
    Setup Data Loaders[Currently supports Cityscapes]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    train_sets = []
    val_sets = []
    val_dataset_names = []


    train_joint_transform_list, train_joint_transform = get_train_joint_transform()
    train_input_transform, val_input_transform = get_input_transforms()
    target_transform, target_train_transform, target_aux_train_transform = get_target_transforms()


    train_set = CityScapesUniform(
        dataroot,
        joint_transform_list=train_joint_transform_list,
        transform=train_input_transform,
        target_transform=target_train_transform,
        class_uniform_pct=0.5,
        class_uniform_tile=1024,
        coarse_boost_classes=None)
    return train_set

