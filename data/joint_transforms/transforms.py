import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional
import torchvision.transforms.functional as F
from enum import Enum
import torch
from torch import Tensor
from PIL import Image

class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]

pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

class JitterRandomCrop(torch.nn.Module):

    def __init__(self, size, scale=(0.5, 1.5), ignore_id=19, input_mean=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if (scale[0] > scale[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ignore_id = ignore_id
        self.input_mean= input_mean

    @staticmethod
    def get_resize_params(img: Tensor, scale: List[float]) -> int:
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        min_value = min(width, height)
        coef = torch.rand(1) * (scale[0] - scale[1]) + scale[1]
        val = int(coef * min_value)
        return val

    @staticmethod
    def get_crop_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def forward(self, data):
        img, lbl = data
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        resize_val = self.get_resize_params(img, self.scale)

        img = F.resize(img, resize_val, pil_modes_mapping[InterpolationMode.BILINEAR])
        lbl = F.resize(lbl, resize_val, pil_modes_mapping[InterpolationMode.NEAREST])

        width, height = F.get_image_size(img)
        # pad the width if needed
        if width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.to_pil_image(img)
            img = F.pad(img, padding, self.input_mean, 'constant')
            img = F.to_tensor(img)
            lbl = F.pad(lbl, padding, self.ignore_id, 'constant')
        # pad the height if needed
        if height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.to_pil_image(img)
            img = F.pad(img, padding, self.input_mean, 'constant')
            img = F.to_tensor(img)
            lbl = F.pad(lbl, padding, self.ignore_id, 'constant')

        i, j, h, w = self.get_crop_params(img, self.size)

        img = F.crop(img, i, j, h, w)
        lbl = F.crop(lbl, i, j, h, w)
        return img, lbl


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img, lbl = data
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            lbl = F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2
            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        img, lbl = data
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)

class JointResize(torch.nn.Module):
    """Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set True for
            ``InterpolationMode.BILINEAR`` only mode.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.

    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, data):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img, lbl = data
        return (F.resize(img, self.size, Image.BILINEAR), F.resize(lbl, self.size, Image.NEAREST))

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)
