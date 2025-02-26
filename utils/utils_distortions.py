import math
import numpy as np
from typing import Union, Tuple
import torch
from torch.nn import functional as F
import scipy


def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a PyTorch tensor to the range [0, 1].

    This function scales the input tensor `x` so that its minimum value becomes 0 and its maximum value becomes 1. If
    the tensor has a constant value (min == max), it is returned unchanged to avoid division by zero.

    Args:
        x (torch.Tensor): The input tensor to be normalized.

    Returns:
        torch.Tensor: The normalized tensor, with values in the range [0, 1].
    """
    minx = x.amin()
    maxx = x.amax()
    diff = maxx - minx
    return (x - minx) / diff.clamp_min(1e-10)


def generate_gaussian_kernel(size: Tuple[int, int], sigma: Union[int, float]) -> torch.Tensor:
    """
    Generates a normalized 2D Gaussian kernel.

    Args:
        size (Tuple[int, int]): Kernel size (height, width).
        sigma (float): Standard deviation of the Gaussian function.

    Returns:
        torch.Tensor: Normalized 2D Gaussian kernel.
    """
    height, width = size
    center_y, center_x = (height -1) /2., (width -1) /2.
    y, x = np.ogrid[-center_y:center_y + 1, -center_x:center_x + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0  # Remove too small values to avoid numerical erros
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh

    return torch.from_numpy(kernel).float()


def generate_disk_kernel(radius: int) -> torch.Tensor:
    """
    Generates a disk-shaped kernel for image filtering.

    Args:
        radius int: Radius of the disk.

    Returns:
        torch.Tensor: a 2D disk kernel normalized so that sum of all elements is 1.
    """
    # Compute the rounded radius value
    rounded_radius = math.ceil(radius - 0.5)

    # Create a grid of x and y coordinates centered at zero
    x, y = np.ogrid[-radius: radius + 1, -radius: radius + 1]

    # Repeat arrays to form a full coordinate grid
    y = np.tile(y.T, y.shape[1])
    x = np.tile(x, x.shape[0]).T

    # Compute absolute values of coordinates
    y, x = np.abs(y), np.abs(x)

    # Compute max and min between x and y
    max_xy = np.maximum(x, y)
    min_xy = np.minimum(x, y)

    # Compute radial distances
    r1 = (radius ** 2 - (max_xy + 0.5) ** 2)
    r2 = (radius ** 2 - (min_xy - 0.5) ** 2)

    # Compute square root values, ensuring non-negative results
    if (r1 > 0).all():
        warn_m1 = r1 ** 0.5
    else:
        warn_m1 = 0

    if (r2 > 0).all():
        warn_m2 = r2 ** 0.5
    else:
        warn_m2 = 0

    # Compute inner and outer boundaries for the disk shape
    m1 = ((radius ** 2 < (max_xy + 0.5) ** 2 + (min_xy - 0.5) ** 2) * (min_xy - 0.5) +
          (radius ** 2 >= (max_xy + 0.5) ** 2 + (min_xy - 0.5) ** 2) * warn_m1)
    m2 = ((radius ** 2 > (max_xy - 0.5) ** 2 + (min_xy + 0.5) ** 2) * (min_xy + 0.5) +
          (radius ** 2 <= (max_xy - 0.5) ** 2 + (min_xy + 0.5) ** 2) * warn_m2)

    # Compute the area of the disk inside each grid cell and apply conditions to refine each grid cell
    disk_area = (radius ** 2 * (0.5 * (np.arcsin(m2 / radius) - np.arcsin(m1 / radius)) +
                                0.25 * (np.sin(2 * np.arcsin(m2 / radius)) - np.sin(2 * np.arcsin(m1 / radius)))) - (
                     max_xy - 0.5) * (m2 - m1) + (m1 - min_xy + 0.5)) * np.logical_or(
        np.logical_and((radius ** 2 < (max_xy + 0.5) ** 2 + (min_xy + 0.5) ** 2),
                       (radius ** 2 > (max_xy - 0.5) ** 2 + (min_xy - 0.5) ** 2)),
        np.logical_and(np.logical_and(min_xy == 0, max_xy - 0.5 < radius), max_xy + 0.5 >= radius))

    disk_area = disk_area + ((max_xy + 0.5) ** 2 + (min_xy + 0.5) ** 2 < radius ** 2)
    disk_area[rounded_radius, rounded_radius] = np.minimum(math.pi * radius ** 2, math.pi / 2)

    # Adjust edge cases when the radius is close to the rounded boundary
    if (rounded_radius > 0) and (radius > rounded_radius - 0.5) and (radius ** 2 < (rounded_radius - 0.5) ** 2 + 0.25):
        m1 = np.sqrt(radius ** 2 - (rounded_radius - 0.5) ** 2)
        m1_normalized = m1 / radius

        sg0 = 2 * (radius ** 2 * (0.5 * np.arcsin(m1_normalized) + 0.25 * np.sin(2 * np.arcsin(m1_normalized))) -
                   m1 * (rounded_radius - 0.5))

        # Apply adjustments to specific positions
        disk_area[2 * rounded_radius, rounded_radius] = sg0
        disk_area[rounded_radius, 2 * rounded_radius] = sg0
        disk_area[rounded_radius, 0] = sg0
        disk_area[0, rounded_radius] = sg0

        disk_area[2 * rounded_radius, rounded_radius] = disk_area[2 * rounded_radius, rounded_radius] - sg0
        disk_area[rounded_radius, 2 * rounded_radius] = disk_area[rounded_radius, 2 * rounded_radius] - sg0
        disk_area[rounded_radius, 2] = disk_area[rounded_radius, 2] - sg0
        disk_area[2, rounded_radius] = disk_area[2, rounded_radius + 1] - sg0

    # Ensure the central pixel does not exceed a value of 1
    disk_area[rounded_radius, rounded_radius] = np.minimum(disk_area[rounded_radius, rounded_radius], 1)

    # Normalize the kernel so that the sum of all values equals 1
    disk_kernel = disk_area / np.sum(disk_area)

    return torch.from_numpy(disk_kernel).float()


def generate_motion_kernel(length: int, angle: int) -> torch.Tensor:
    """
    Generates a motion blur kernel.

    Args:
        length (int): the length of the motion blur.
        angle (int): the angle of motion blur in degrees.
    Returns:
        torch.Tensor: a motion blur kernel as a tensor.
    """
    eps = 2.2204e-16
    length = max(1, length)
    half_len = (length - 1) / 2.
    phi = (angle % 180) / 180 * math.pi

    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    x_sign = np.sign(cos_phi)
    line_width = 1

    # Determine grid dimensions
    grid_x = int(half_len * cos_phi + line_width * x_sign - length * eps)
    grid_y = int(half_len * sin_phi + line_width - length * eps)

    # Create coordinate mesh grid
    x, y = np.mgrid[0:grid_x + (1 * x_sign):x_sign, 0:grid_y + 1]
    x, y = x.T, y.T

    # Compute distance of each point from the motion line
    dist_to_line = y * cos_phi - x * sin_phi
    radius = (x ** 2 + y ** 2) ** 0.5

    # Find pixels representing the motion boundary
    last_pixels = np.where(np.logical_and((radius >= half_len), (abs(dist_to_line) <= line_width)))
    x_shift_last_pixels = half_len - np.abs((x[last_pixels] + dist_to_line[last_pixels] * sin_phi) / cos_phi)

    # Adjust distance values
    dist_to_line[last_pixels] = np.sqrt(dist_to_line[last_pixels] ** 2 + x_shift_last_pixels ** 2)
    dist_to_line = line_width + eps - np.abs(dist_to_line)
    dist_to_line[dist_to_line < 0] = 0

    # Create motion blur kernel and mirror it for symmetry
    motion_kernel = np.rot90(dist_to_line, 2)
    temp_kernel = np.zeros((motion_kernel.shape[0] * 2 - 1, motion_kernel.shape[1] * 2 - 1))
    temp_kernel[0:motion_kernel.shape[0], 0:motion_kernel.shape[1]] = motion_kernel
    temp_kernel[(motion_kernel.shape[0]) - 1:, motion_kernel.shape[1] - 1:] = dist_to_line
    motion_kernel = temp_kernel

    motion_kernel /= np.sum(motion_kernel) + eps * length * length

    if cos_phi > 0:
        motion_kernel = np.flipud(motion_kernel)

    return torch.from_numpy(motion_kernel).float()


def filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    img = img.float()
    k1 = kernel.size(-2)
    k2 = kernel.size(-1)

    b, c, h, w = img.size()
    if k1 % 2 == 1 or k2 % 2 == 1:
        img = F.pad(img, (k2 // 2, k2 // 2, k1 // 2, k1 // 2), mode='replicate')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k1, k2)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k1, k2).repeat(1, c, 1, 1).view(b * c, 1, k1, k2)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def curves(xx: torch.Tensor, coef: float | list) -> torch.Tensor:
    """
    Applies a non-linear transformation to a tensor using cubic spline interpolation.

    This function maps the values in the input tensor `xx` to new values based on a
    cubic spline defined by control points derived from the parameter `coef`. The
    output values are clamped to the range [0, 1].

    Args:
        xx (torch.Tensor): The input tensor to be transformed. Values are expected
            to be in the range [0, 1].
        coef (float or list): The control points for the spline:
            - If `coef` is a float, it defines a simple curve with default control
              points [0.5] and [coef].
            - If `coef` is a list of two floats [a, b], these values are used as
              control points for the curve.

    Returns:
        torch.Tensor: A tensor with the same shape as `xx`, where each value is
        transformed based on the spline curve and clamped to the range [0, 1].

    Raises:
        ValueError: If `coef` is not a float or a list of length 2.
    """
    if type(coef) == list:
        coef = [[0.3, 0.5, 0.7],
                [coef[0], 0.5, coef[1]]]
    else:
        coef = [[0.5], [coef]]

    x = np.array([0] + [p for p in coef[0]] + [1])
    y = np.array([0] + [p for p in coef[1]] + [1])

    cs = spline(x, y)

    yy = ppval(cs, xx)

    yy = torch.clamp(yy, 0, 1)

    return yy


def spline(x: np.ndarray, y: np.ndarray) -> tuple:
    """
       Computes the coefficients for a cubic spline interpolation based on the input points.

       This function calculates the cubic spline coefficients that define a smooth curve
       passing through the input points `x` and `y`. It handles cases where the number
       of points is exactly 3 or greater than 3, leveraging tridiagonal systems for
       efficient computation.

       Args:
           x (np.ndarray): A 1D array of input points (independent variable) in ascending order.
                           Must have at least 3 points.
           y (np.ndarray): A 1D array of values (dependent variable) corresponding to the points in `x`.

       Returns:
           tuple: A tuple `pp` containing:
               - x (np.ndarray): The input points used for the spline.
               - coefs (np.ndarray): The coefficients of the cubic spline for each interval.
               - n (int): The number of points in `x`.
               - l (int): Number of intervals (equal to `n-1`).
               - d (int): Dimension of the data (always 1 for scalar input).

       Raises:
           ValueError: If `x.shape[0]` (number of points in `x`) is less than 3.
    """
    n = x.shape[0]
    dd = 1
    dx = np.diff(x)
    divdif = np.diff(y) / dx

    if n == 3:
        y[1:3] = divdif
        y[2] = np.diff(divdif.T).T / (x[2] - x[0])
        y[1] -= y[2] * dx[0]
        dlk = y[[2, 1, 0]].shape[0]
        l = x[[0, 2]].shape[0] - 1
        dl = np.prod(dd) * l
        k = np.fix(dlk / dl + 100 * 2.2204e-16)

        pp = (x[[0, 2]], y[[2, 1, 0]], l, int(k), dd)

    elif n > 3:
        b = np.zeros(n)
        b[1:n - 1] = 3 * (dx[1:n] * divdif[0:n - 2] + dx[0:n - 2] * divdif[1:n])

        x31 = x[2] - x[0]
        xn = x[n - 1] - x[n - 3]

        b[0] = ((dx[0] + 2 * x31) * dx[1] * divdif[0] + dx[0] ** 2 * divdif[1]) / x31
        b[n - 1] = (dx[n - 2] ** 2 * divdif[n - 3] + (2 * xn + dx[n - 2]) * dx[n - 3] * divdif[n - 2]) / xn;

        dxt = dx.T
        c = np.zeros((3, 5))
        c[0, :] = [x31] + list(dxt[0:n - 2]) + [0]
        c[1, :] = [dxt[1]] + list(2 * (dxt[1:n - 1] + dxt[0:n - 2])) + [dxt[n - 3]]
        c[2, :] = [0] + list(dxt[1:n - 1]) + [xn]

        c = scipy.sparse.dia_matrix((c, [-1, 0, 1]), shape=(5, 5))
        c = scipy.sparse.csc_matrix(c)
        ic = scipy.sparse.linalg.inv(c)
        s = b * ic

        n = x.shape[0]
        d = 1
        dxd = dx

        dzzdx = (divdif - s[0:n - 1]) / dxd
        dzdxdx = (s[1:n] - divdif) / dxd

        coefs = np.vstack(((dzdxdx - dzzdx) / dxd, 2 * dzzdx - dzdxdx, s[0:n - 1], y[0:n - 1])).T

        pp = (x, coefs, x.shape[0], x.shape[0], d)
    else:
        raise ValueError('x.shape[0] must be >= 3')

    return pp


def ppval(pp: np.ndarray, xx: torch.Tensor) -> torch.Tensor:
    """
       Evaluates a piecewise polynomial (spline) at specified points.

       This function takes a piecewise polynomial representation (`pp`) and evaluates
       the spline at points specified in the tensor `xx`. It uses the coefficients
       of the spline and the interval structure stored in `pp` to compute the values.

       Args:
           pp (np.ndarray): The piecewise polynomial representation, typically as a tuple:
                            - `b` (np.ndarray): Breakpoints of the spline intervals.
                            - `c` (np.ndarray): Coefficients of the spline polynomials.
                            - `l` (int): Number of intervals (equal to `len(b) - 1`).
                            - `k` (int): Polynomial degree plus one (e.g., `k=4` for cubic splines).
                            - `dd` (int): Dimension of the polynomial (1 for scalar values).
           xx (torch.Tensor): A tensor of points at which to evaluate the spline.

       Returns:
           torch.Tensor: A tensor of the same shape as `xx`, containing the evaluated values of the spline.
    """
    lx = torch.numel(xx)
    xs = xx.reshape(1, lx)
    b, c, l, k, dd = pp
    b = torch.as_tensor(b, device=xx.device)
    ranges = b.clone()
    ranges[0] = -torch.inf
    ranges[-1] = torch.inf
    index = histc(xs, ranges)

    xs = xs - b[index]

    c = torch.as_tensor(c, device=xx.device)

    if len(c.shape) == 1:
        v = c[0]
        for i in range(1, k):
            v = xs * v + c[i]
    else:
        v = c[index, 0]

        for i in range(1, k - 1):
            v = xs * v + c[index, i]
    v = v.view(xx.shape)
    return v


def histc(x: torch.Tensor, binranges: torch.Tensor) -> torch.Tensor:
    """
    Categorizes values in a tensor into specified bin ranges.

    This function takes a tensor `x` and a tensor `binranges`, and assigns each value in `x`
    to the index of the interval in `binranges` that contains it. The output is a tensor of
    indices corresponding to the intervals.

    Args:
        x (torch.Tensor): A tensor of values to be categorized into bins.
        binranges (torch.Tensor): A tensor defining the edges of the bins.
                                  Must be sorted in ascending order.

    Returns:
        torch.Tensor: A tensor of indices, where each index represents the bin
                      (from `binranges`) to which the corresponding value in `x` belongs.
    """
    indices = torch.bucketize(x, binranges)
    return torch.remainder(indices, len(binranges)) - 1


def imscatter(x: torch.Tensor, amount: float, iterations=1) -> torch.Tensor:
    """
    Applies a random pixel displacement to an image, creating a "scatter" effect.

    This function takes an input tensor representing an image and displaces its pixels
    randomly based on a specified magnitude (`amount`). The scattering can be repeated
    for multiple iterations to amplify the effect.

    Args:
        x (torch.Tensor): A 3D tensor representing the input image with shape
                          (C, H, W), where C is the number of channels, H is the height,
                          and W is the width.
        amount (float): The magnitude of the random displacements applied to the pixels.
        iterations (int, optional): The number of iterations of scattering to perform.
                                    Default is 1.

    Returns:
        torch.Tensor: A 3D tensor representing the scattered image with the same shape
                      as the input (C, H, W).
    """
    y = x
    for i in range(iterations):
        shiftmap = torch.randn((2, x.shape[1], x.shape[2]), device=x.device) * amount

        sy = shiftmap[0, :, :]
        sx = shiftmap[1, :, :]

        m_sx = torch.ceil(torch.abs(torch.max(sx))).to(torch.int32)
        m_sy = torch.ceil(torch.abs(torch.max(sy))).to(torch.int32)

        y = F.pad(y, (m_sy, m_sy), mode='replicate')
        y = F.pad(y.transpose(2, 1), (m_sx, m_sx), mode='replicate').transpose(2, 1)

        sy = F.pad(sy, (m_sy, m_sy), mode='replicate')
        sy = F.pad(sy.transpose(1, 0), (m_sx, m_sx), mode='replicate').transpose(1, 0)
        sx = F.pad(sx, (m_sy, m_sy), mode='replicate')
        sx = F.pad(sx.transpose(1, 0), (m_sx, m_sx), mode='replicate').transpose(1, 0)

        xx, yy = torch.as_tensor(np.mgrid[0:y.shape[1], 0:y.shape[2]], device=x.device)

        z = torch.zeros_like(y)
        bx = (xx - sx)
        by = (yy - sy)
        for i in range(3):
            j = bilinear_interpolate_torch(y[i, ...], by, bx)
            z[i, :, :] = j

        y = z[:, m_sy:m_sy + x.shape[1], m_sx:m_sx + x.shape[2]]
    return y


def bilinear_interpolate_torch(im: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
       Performs bilinear interpolation for the given coordinates on a 2D tensor.

       This function computes interpolated values at the given (x, y) coordinates
       on the input tensor `im` using the bilinear interpolation method. The result
       is a smooth approximation of values in positions between the discrete grid points.

       Args:
           im (torch.Tensor): A 2D tensor (H, W) representing the input image or grid
                              from which values will be interpolated.
           x (torch.Tensor): A tensor of x-coordinates where interpolation should be performed.
                             Must have the same shape as `y`.
           y (torch.Tensor): A tensor of y-coordinates where interpolation should be performed.
                             Must have the same shape as `x`.
           eps (float, optional): A small constant added to avoid division by zero during
                                  interpolation. Default is 1e-8.

       Returns:
           torch.Tensor: A tensor containing the interpolated values at the specified (x, y)
                         coordinates. The output shape matches the shape of the input `x` and `y`.
    """
    dtype_long = torch.LongTensor

    x0 = torch.floor(x).type(dtype_long).to(im.device)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long).to(im.device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    R1 = Ia * (x1 - x) / (x1 - x0 + eps) + Ic * (x - x0) / (x1 - x0 + eps)
    R2 = Ib * (x1 - x) / (x1 - x0 + eps) + Id * (x - x0) / (x1 - x0 + eps)
    P = R1 * (y1 - y) / (y1 - y0 + eps) + R2 * (y - y0) / (y1 - y0 + eps)
    return P
