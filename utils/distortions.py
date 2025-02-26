import torch
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

from utils.utils_distortions import (generate_gaussian_kernel, generate_disk_kernel, generate_motion_kernel, filter2D,
                                     curves, imscatter, normalize)

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

if os.name == 'posix':
    DITHER_FILE = "dither.so"
elif os.name == 'nt':
    DITHER_FILE = "dither.dll"
else:
    raise NameError("Uknown OS")

DITHER_CPP = ctypes.CDLL(str(PROJECT_ROOT / "utils" / "dither_extension" / DITHER_FILE))
DITHER_CPP.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                       ctypes.c_int]


"""
The distortions functions implemented in this file follows the distortions proposed in the ARNIQA paper 
https://arxiv.org/abs/2310.14918.
"""


def gaussian_blur(x: torch.Tensor, blur_sigma: float = 0.1) -> torch.Tensor:
    """
    Applies a Gaussian blur to an input tensor.

    Args:
        x (torch.Tensor): The input tensor, expected to have shape
                          (C, H, W) for a single image or (N, C, H, W) for a batch of images.
                          If (C, H, W) is provided, it will be unsqueezed to (1, C, H, W).
        blur_sigma (float, optional): The standard deviation of the Gaussian kernel. Determines
                                    the amount of blur. Default is 0.1.

    Returns:
        torch.Tensor: The blurred tensor with the same shape as the input tensor.
    """
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    gaussian_kernel = generate_gaussian_kernel((fs, fs), blur_sigma)

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, gaussian_kernel.unsqueeze(0)).squeeze(0)
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
    disk_kernel = generate_disk_kernel(radius)

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, disk_kernel.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(image: torch.Tensor, length: int, angle: bool = None) -> torch.Tensor:
    """
    Applies a motion blur effect to an input tensor using a linear motion kernel.

    Args:
        image (torch.Tensor): The input tensor, expected to have shape
                          (C, H, W) for a single image or (N, C, H, W) for a batch of images.
                          If (C, H, W) is provided, it will be unsqueezed to (1, C, H, W).
        length (int): The length of the motion blur kernel, which determines the intensity
                      and spread of the blur effect.
        angle (bool, optional): The angle of the motion blur in degrees. If not provided,
                                a random angle between 0 and 180 degrees will be used.

    Returns:
        torch.Tensor: The motion-blurred tensor with the same shape as the input tensor.
    """
    if angle is None:
        angle = random.randint(0, 180)
    motion_kernel = generate_motion_kernel(length, angle)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    dist_image = filter2D(image, motion_kernel.unsqueeze(0)).squeeze(0)
    return dist_image


def color_diffusion(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Applies a color diffusion effect to an input image by blurring and scaling its color channels
    in the Lab color space.

    Args:
        x (torch.Tensor): The input tensor representing an image, expected to have shape
                          (C, H, W) where C=3 for RGB channels. The channels are assumed to be
                          in RGB order.
        factor (int): The intensity of the color diffusion effect. Higher values result in
                      stronger blurring and scaling of the color channels.

    Returns:
        torch.Tensor: The image tensor with the color diffusion effect applied, in RGB format
                      with shape (C, H, W) where C=3.
    """
    blur_sigma = 1.5 * factor + 2
    scaling = factor
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)

    fs = 2 * math.ceil(2 * blur_sigma) + 1
    gaussian_kernel = generate_gaussian_kernel((fs, fs), blur_sigma)

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    diff_ab = filter2D(lab[:, 1:3, ...], gaussian_kernel.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling

    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def color_shift(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Applies a gradient-guided color shift effect to an input image. This operation displaces
    a specific color channel of the image spatially while blending it with the original
    image based on edge information from the gradient map.

    Args:
        x (torch.Tensor): The input image tensor with shape (C, H, W), where C=3 for RGB channels.
                          The tensor is expected to be normalized in the range [0, 1].
        factor (int): The magnitude of the spatial shift applied to the selected color channel.
                      Higher values result in a more noticeable shift.

    Returns:
        torch.Tensor: The modified image tensor with the color shift effect applied, maintaining
                      the original shape (C, H, W).
    """
    def perc(img_tensor: torch.Tensor, percentile: int) -> torch.Tensor:
        """
          Computes the given percentile value from a tensor.

          Args:
              img_tensor (torch.Tensor): The input tensor containing numerical values.
              percentile (float): The percentile to compute, in the range [0, 100].

          Returns:
              torch.Tensor: The value at the specified percentile.
        """
        xs = torch.sort(img_tensor)
        i = len(xs) * percentile / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    gaussian_kernel = generate_gaussian_kernel((fs, fs), 4)

    e = filter2D(e, gaussian_kernel.unsqueeze(0))

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
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * factor)[0].astype(int)

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


def color_saturation_hsv(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjusts the color saturation of an RGB image by scaling the saturation channel in HSV color space.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C = 3` (RGB format) and pixel values in the range `[0, 1]`.
    factor : float
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


def color_saturation_lab(x: torch.Tensor, factor: int) -> torch.Tensor:
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


def compress_jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:
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


def compress_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
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


def white_noise(x: torch.Tensor, variance: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    """
    Adds Gaussian white noise to an input image tensor and optionally applies clipping and rounding.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` or `(N, C, H, W)` where `C` is the number of channels.
        Pixel values should be in the range `[0, 1]`.
    variance : float
        Variance of the Gaussian white noise to be added.
    clip : bool, optional
        If `True`, clips the resulting tensor to the range `[0, 1]`. Default is `True`.
    rounds : bool, optional
        If `True`, rounds the resulting tensor to the nearest 8-bit integer (simulates quantization). Default is `False`.

    Returns:
    torch.Tensor
        Output tensor with added Gaussian noise and optionally clipped or rounded values.
        The output shape matches the input shape.
    """
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(variance)

    y = x + noise

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.
    return y


def white_noise_ycbcr(x: torch.Tensor, variance: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    """
        Adds Gaussian white noise to an input image tensor in the YCbCr color space
        and optionally applies clipping and rounding after converting back to RGB.

        Args:
        x : torch.Tensor
            Input image tensor with shape `(C, H, W)` or `(N, C, H, W)` where `C` is the number of channels.
            Pixel values should be in the range `[0, 1]`.
        variance : float
            Variance of the Gaussian white noise to be added.
        clip : bool, optional
            If `True`, clips the resulting tensor to the range `[0, 1]`. Default is `True`.
        rounds : bool, optional
            If `True`, rounds the resulting tensor to the nearest 8-bit integer (simulates quantization). Default is `False`.

        Returns:
        torch.Tensor
            Output tensor with added Gaussian noise, converted back to RGB, and optionally clipped or rounded.
            The output shape matches the input shape.
    """
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(variance)

    ycbcr = kornia.color.rgb_to_ycbcr(x)
    y = ycbcr + noise

    y = kornia.color.ycbcr_to_rgb(y)

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.

    return y


def impulse_noise(x: torch.Tensor, noise_density: float, sp_ratio: float = 0.5) -> torch.Tensor:
    """
      Adds impulse noise (salt-and-pepper noise) to an input image tensor.

      Args:
      x : torch.Tensor
          Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
          Pixel values should be in the range `[0, 1]`.
      noise_density : float
          Density of the impulse noise. This value determines the proportion of pixels
          affected by salt-and-pepper noise. Should be in the range `[0, 1]`.
      sp_ratio : float, optional
          Salt-to-pepper ratio. Determines the proportion of affected pixels that become
          "salt" (white, value = 1) vs "pepper" (black, value = 0). Default is `0.5` (equal salt and pepper).

      Returns:
      torch.Tensor
          Output tensor with added salt-and-pepper noise. The shape matches the input shape.
    """
    num_sp = int(noise_density * x.shape[0] * x.shape[1] * x.shape[2])

    coords = np.concatenate((np.random.randint(0, 3, (num_sp, 1)),
                             np.random.randint(0, x.shape[1], (num_sp, 1)),
                             np.random.randint(0, x.shape[2], (num_sp, 1))), 1)

    num_salt = int(sp_ratio * num_sp)

    coords_salt = coords[:num_salt].transpose(1, 0)
    coords_pepper = coords[num_salt:].transpose(1, 0)
    x[*coords_salt] = 1
    x[*coords_pepper] = 0

    return x


def multiplicative_noise(x: torch.Tensor, variance: float) -> torch.Tensor:
    """
        Adds multiplicative noise to an input image tensor.

        Args:
        x : torch.Tensor
            Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
            Pixel values should be in the range `[0, 1]`.
        variance : float
            Variance of the multiplicative noise. Determines the intensity of the noise.

        Returns:
        torch.Tensor
            Output tensor with added multiplicative noise. Pixel values are clipped to the range `[0, 1]`.
    """
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(variance)
    y = x + x * noise
    y = torch.clip(y, 0, 1)
    return y


def brighten(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjusts the brightness of an input image tensor by modifying its luminance and applying a tone curve.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
        Pixel values should be in the range `[0, 1]` and represent an RGB image.
    factor : float
        Brightness adjustment factor. Positive values increase brightness,
        while negative values decrease brightness.

    Returns:
    torch.Tensor
        Output image tensor with adjusted brightness. Pixel values remain in the range `[0, 1]`.
    """
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)

    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + factor / 2)
    lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 + factor / 2)

    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)

    y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def darken(x: torch.Tensor, factor: float, use_lab: bool = False) -> torch.Tensor:
    """
       Reduces the brightness of an input image tensor by applying a tone curve and optionally adjusting luminance in LAB color space.

       Args:
       -----
       x : torch.Tensor
           Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
           Pixel values should be in the range `[0, 1]` and represent an RGB image.
       factor : float
           Darkness adjustment factor. Positive values increase the level of darkening.
       use_lab : bool, optional
           If `True`, adjusts the luminance in the LAB color space in addition to applying the tone curve. Default is `False`.

       Returns:
       --------
       torch.Tensor
           Output image tensor with reduced brightness. Pixel values remain in the range `[0, 1]`.
       """
    x = x[[2, 1, 0], :, :]
    lab = kornia.color.rgb_to_lab(x)
    if use_lab:
        l = lab[0, ...] / 100.
        l_ = curves(l, 0.5 + factor / 2)
        lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 - factor / 2)

    if use_lab:
        j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
        y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def mean_shift(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjusts the mean intensity of an image by adding a constant value to its pixel intensities.

    Args:
    -----
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
        Pixel values should be in the range `[0, 1]` and represent an RGB image.
    factor : float
        The value to be added to the pixel intensities. Positive values brighten the image,
        while negative values darken it.

    Returns:
    --------
    torch.Tensor
        Output image tensor with adjusted mean intensity. Pixel values are clamped to the range `[0, 1]`.
    """
    x = x[[2, 1, 0], :, :]

    y = torch.clamp(x + factor, 0, 1)
    return y[[2, 1, 0]]


def jitter(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
       Applies a jitter effect to an image by randomly displacing its pixels.

       Args:
       x : torch.Tensor
           Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
           Pixel values should be in the range `[0, 1]` and represent an RGB image.
       factor : float
           The magnitude of pixel displacement. Larger values result in more noticeable jitter.

       Returns:
       torch.Tensor
           Output image tensor with the applied jitter effect. Pixel values remain in the range `[0, 1]`.
       """
    y = imscatter(x, factor, 5)
    return y


def non_eccentricity_patch(x: torch.Tensor, num_patches: int) -> torch.Tensor:
    """
     Applies a non-eccentricity patch effect by copying random patches of the image to nearby locations.

     Args:
     x : torch.Tensor
         Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
         Pixel values should be in the range `[0, 1]`.
     num_pacthes : int
         Number of patches to generate and apply.

     Returns:
     torch.Tensor
         Output image tensor with the applied non-eccentricity patch effect. Pixel values remain in the range `[0, 1]`.

     Notes:
     - The function randomly selects patches of size `16x16` pixels from the input image and copies them
       to nearby locations within a `16-pixel` radius.
     - The original image content outside the patches remains unchanged.
     """
    patch_size = [16, 16]
    radius = 16
    h_min = radius
    w_min = radius
    c, h, w = x.shape

    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius

    for i in range(num_patches):
        w_start = round(random.random() * (w_max - w_min)) + w_min
        h_start = round(random.random() * (h_max - h_min)) + h_min
        patch = x[:, h_start:h_start + patch_size[0], w_start:w_start + patch_size[0]]

        rand_w_start = round((random.random() - 0.5) * radius + w_start)
        rand_h_start = round((random.random() - 0.5) * radius + h_start)
        x[:, rand_h_start:rand_h_start + patch_size[0], rand_w_start:rand_w_start + patch_size[0]] = patch

    return x


def pixelate(x: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Applies a pixelation effect to the input image by resizing and then resizing it back.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
        Pixel values should be in the range `[0, 1]`.
    strength : float
        Strength of the pixelation effect. Higher values result in stronger pixelation.

    Returns:
    torch.Tensor
        Output image tensor with the pixelation effect applied. Pixel values remain in the range `[0, 1]`.
    """
    z = 0.95 - strength ** 0.6
    c, h, w = x.shape

    ylo = kornia.geometry.transform.resize(x, (int(h * z), int(w * z)), 'nearest')
    y = kornia.geometry.transform.resize(ylo, (h, w), 'nearest')

    return y


def quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
    """
    Applies quantization to the input image by reducing the number of intensity levels.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
        Pixel values should be in the range `[0, 1]`.
    levels : int
        The number of intensity levels to quantize the image into. Must be between 1 and 256.

    Returns:
    torch.Tensor
        The quantized image tensor with reduced intensity levels. Pixel values remain in the range `[0, 1]`.
    """
    image = kornia.color.rgb_to_grayscale(x) * 255
    image = image.cpu().numpy()
    num_classes = levels

    # minimum variance thresholding
    hist, bins = np.histogram(image, num_classes, [0, 255])

    return_thresholds = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        return_thresholds[i] = bins[i + 1]

    # quantize image with thresholds
    bins = torch.tensor([0] + return_thresholds.tolist() + [256])
    bins = bins.type(torch.int)
    image = torch.bucketize(x.contiguous() * 255., bins).to(torch.float32)
    image = normalize(image)
    return image


def color_block(x: torch.Tensor, num_patches: int) -> torch.Tensor:
    """
    Adds color blocks to the input image by randomly placing patches of a uniform color.

    Args:
    x : torch.Tensor
        Input image tensor with shape `(C, H, W)` where `C` is the number of channels.
        The pixel values should be in the range `[0, 1]`.
    pnum : int
        The number of color blocks (patches) to add to the image.

    Returns:
    torch.Tensor
        The image with added color blocks, with the same shape as the input image.
    """
    patch_size = [32, 32]

    c, w, h = x.shape

    y = x

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    for i in range(num_patches):
        color = np.random.random(3)
        px = math.floor(random.random() * w_max)
        py = math.floor(random.random() * h_max)
        patch = torch.ones((3, patch_size[0], patch_size[1]))
        for j in range(3):
            patch[j, ...] *= color[j]
        y[:, px:px + patch_size[0], py:py + patch_size[1]] = patch

    return y


def high_sharpen(x: torch.Tensor, factor: int, radius: int = 3) -> torch.Tensor:
    """
    Applies a high-pass sharpening filter to enhance the high-frequency details of an image.

    Args:
    x : torch.Tensor
        Input image tensor in the format `(C, H, W)` where `C` is the number of channels (RGB),
        with pixel values in the range `[0, 1]`.
    factor : int
        The intensity of sharpening to apply. A higher value results in stronger sharpening.
    radius : int, optional (default=3)
        The radius of the Gaussian blur used to create the sharpening filter. A larger radius
        applies a broader blur before the sharpening step.

    Returns:
    torch.Tensor
        The sharpened image tensor with the same shape as the input.
     """
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0:1, ...].unsqueeze(0)

    filt_radius = math.ceil(radius * 2)
    fs = 2 * filt_radius + 1
    gaussian_kernel = generate_gaussian_kernel((fs, fs), filt_radius)

    sharp_filter = torch.zeros((fs, fs))
    sharp_filter[filt_radius, filt_radius] = 1
    sharp_filter = sharp_filter - gaussian_kernel

    sharp_filter *= factor
    sharp_filter[filt_radius, filt_radius] += 1

    l = filter2D(l, sharp_filter.unsqueeze(0))

    lab[0, ...] = l

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    y = kornia.color.lab_to_rgb(lab)
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def linear_contrast_change(x: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Applies a linear contrast change to an image.

    Args:
    x : torch.Tensor
        Input image tensor in the format `(C, H, W)` where `C` is the number of channels (RGB),
        with pixel values in the range `[0, 1]`.
    factor : float
        The amount of contrast change to apply. A positive value increases contrast, while a negative value decreases it.

    Returns:
    torch.Tensor
        The contrast-adjusted image tensor with the same shape as the input.
    """
    y = curves(x, [0.25 - factor / 4, 0.75 + factor / 4])
    return y


def non_linear_contrast_change(x: torch.Tensor, output_offset_value: float, output_central_value: float = 0.5,
                               input_offset_value: float = 0.5, input_central_value: float = 0.5) -> torch.Tensor:
    """
      Applies a non-linear contrast adjustment to an image.

      Args:
      x : torch.Tensor
          Input image tensor in the format `(C, H, W)` where `C` is the number of channels (e.g., RGB),
          with pixel values in the range `[0, 1]`.
      output_offset_value : float
          The offset value for the output range, which determines the contrast intensity in the output image.
      output_central_value : float, optional, default=0.5
          The central value for the output range. This is where the output image's pixel values are centered.
      input_offset_value : float, optional, default=0.5
          The offset value for the input range. This defines the range of input values to be adjusted.
      input_central_value : float, optional, default=0.5
          The central value for the input range. This is where the input image's pixel values are centered.

      Returns:
      torch.Tensor
          The contrast-adjusted image tensor with the same shape as the input.
      """
    low_in = input_central_value - input_offset_value
    high_in = input_central_value + input_offset_value
    low_out = output_central_value - output_offset_value
    high_out = output_central_value + output_offset_value

    # Clip the input image to the specified input range
    x = np.clip(x, low_in, high_in)

    # Calculate the slope and intercept of the linear transformation
    slope = (high_out - low_out) / (high_in - low_in)
    intercept = low_out - slope * low_in

    # Apply the linear transformation to adjust the pixel values
    y = slope * x + intercept

    # Clip the adjusted image to the specified output range
    y = np.clip(y, low_out, high_out)

    return y
