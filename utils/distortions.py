import torch
from torchaudio.functional import dither
from torchvision.io.image import decode_jpeg, encode_jpeg
from torchvision import transforms
import numpy as np
import random
import math
from torch.nn import functional as F
from pathlib import Path
import io
import os
from PIL import Image
import ctypes
import kornia

from utils.utils_distortions import fspecial, filter2D, curves, imscatter, mapmm

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

if os.name == 'posix':
    dither_file = "dither.so"
elif os.name == 'nt':
    dither_file = "dither.dll"
else:
    raise NameError("Uknown OS")

dither_cpp = ctypes.CDLL(str(PROJECT_ROOT / "utils" / "dither_extension" / dither_file))
dither_cpp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                       ctypes.c_int]


def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    """
    Applies a Gaussian blur to an input tensor.

    Args:
        x (torch.Tensor): The input tensor, expected to have shape
                          (C, H, W) for a single image or (N, C, H, W) for a batch of images.
                          If (C, H, W) is provided, it will be unsqueezed to (1, C, H, W).
        blur_sigma (int, optional): The standard deviation of the Gaussian kernel. Determines
                                    the amount of blur. Default is 0.1.

    Returns:
        torch.Tensor: The blurred tensor with the same shape as the input tensor.
    """
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
     Applies a lens blur effect to an input tensor using a disk-shaped kernel.

     Args:
         x (torch.Tensor): The input tensor, expected to have shape
                           (C, H, W) for a single image or (N, C, H, W) for a batch of images.
                           If (C, H, W) is provided, it will be unsqueezed to (1, C, H, W).
         radius (int): The radius of the disk-shaped kernel, which determines the intensity
                       and spread of the blur effect.

     Returns:
         torch.Tensor: The blurred tensor with the same shape as the input tensor.
    """
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    """
    Applies a motion blur effect to an input tensor using a linear motion kernel.

    Args:
        x (torch.Tensor): The input tensor, expected to have shape
                          (C, H, W) for a single image or (N, C, H, W) for a batch of images.
                          If (C, H, W) is provided, it will be unsqueezed to (1, C, H, W).
        radius (int): The length of the motion blur kernel, which determines the intensity
                      and spread of the blur effect.
        angle (bool, optional): The angle of the motion blur in degrees. If not provided,
                                a random angle between 0 and 180 degrees will be used.

    Returns:
        torch.Tensor: The motion-blurred tensor with the same shape as the input tensor.
    """
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def color_diffusion(x: torch.Tensor, amount: int) -> torch.Tensor:
    """
    Applies a color diffusion effect to an input image by blurring and scaling its color channels
    in the Lab color space.

    Args:
        x (torch.Tensor): The input tensor representing an image, expected to have shape
                          (C, H, W) where C=3 for RGB channels. The channels are assumed to be
                          in RGB order.
        amount (int): The intensity of the color diffusion effect. Higher values result in
                      stronger blurring and scaling of the color channels.

    Returns:
        torch.Tensor: The image tensor with the color diffusion effect applied, in RGB format
                      with shape (C, H, W) where C=3.
    """
    blur_sigma = 1.5 * amount + 2
    scaling = amount
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)

    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    diff_ab = filter2D(lab[:, 1:3, ...], h.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling

    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    """
    Applies a gradient-guided color shift effect to an input image. This operation displaces
    a specific color channel of the image spatially while blending it with the original
    image based on edge information from the gradient map.

    Args:
        x (torch.Tensor): The input image tensor with shape (C, H, W), where C=3 for RGB channels.
                          The tensor is expected to be normalized in the range [0, 1].
        amount (int): The magnitude of the spatial shift applied to the selected color channel.
                      Higher values result in a more noticeable shift.

    Returns:
        torch.Tensor: The modified image tensor with the color shift effect applied, maintaining
                      the original shape (C, H, W).
    """
    def perc(x, perc):
        xs = torch.sort(x)
        i = len(xs) * perc / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float()

    e = filter2D(e, h.unsqueeze(0))

    mine = torch.min(e)
    maxe = torch.max(e)

    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])

    e = torch.max(torch.min(e, valuehi), valuelo)

    channel = 1
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x


def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Adjusts the color saturation of an RGB image by scaling the saturation channel in HSV color space.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C = 3` (RGB format) and pixel values in the range `[0, 1]`.
    factor : int
        Scaling factor to adjust the saturation. A value greater than 1 increases saturation,
        while a value between 0 and 1 decreases saturation.

    Returns:
    torch.Tensor
        Output image tensor with the same shape `(C, H, W)` as the input, but with adjusted saturation.
    """
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Adjusts the color saturation of an RGB image by scaling the chromatic channels in LAB color space.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C = 3` (RGB format) and pixel values in the range `[0, 1]`.
    factor : int
        Scaling factor to adjust the saturation. A value greater than 1 increases saturation,
        while a value between 0 and 1 decreases saturation.

    Returns:
    torch.Tensor
        Output image tensor with the same shape `(C, H, W)` as the input, but with adjusted saturation.
    """
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """
        Applies JPEG2000 compression to an input image tensor and returns the compressed and decompressed image.

        Args:
        x : torch.Tensor
            Input image tensor with shape `(C, H, W)` where `C = 3` (RGB format) and pixel values in the range `[0, 1]`.
        ratio : int
            Compression ratio for the JPEG2000 format. Lower values result in higher compression (and lower quality).

        Returns:
        torch.Tensor
            Output image tensor with the same shape `(C, H, W)` as the input, after being compressed and decompressed.
    """
    ratio = int(ratio)
    compression_params = {
        'quality_mode': 'rates',
        'quality_layers': [ratio],  # Compression ratio
        'num_resolutions': 8,  # Number of wavelet decompositions
        'prog_order': 'LRCP',  # Progression order: Layer-Resolution-Component-Position
    }

    # Compress the image and save it using the JPEG2000 format
    x *= 255.
    x = x.byte().cpu().numpy()

    x = Image.fromarray(x.transpose(1, 2, 0), 'RGB')

    with io.BytesIO() as output:
        x.save(output, format='JPEG2000', **compression_params)
        compressed_data = output.getvalue()

    y = Image.open(io.BytesIO(compressed_data))
    y = transforms.ToTensor()(y)

    return y


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Applies JPEG compression to an input image tensor and returns the compressed and decompressed image.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C = 3` (RGB format) and pixel values in the range `[0, 1]`.
    quality : int
        Compression quality for the JPEG format, specified as an integer between 0 and 100.
        Higher values result in better quality and lower compression.

    Returns:
    torch.Tensor
        Output image tensor with the same shape `(C, H, W)` as the input, after being compressed and decompressed.
    """
    x *= 255.
    y = encode_jpeg(x.byte().cpu(), quality=quality)
    y = (decode_jpeg(y) / 255.).to(torch.float32)
    return y
