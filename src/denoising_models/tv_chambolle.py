"""Implements Chambolle's projection algorithm for total variation image denoising. See https://www.ipol.im/pub/art/2013/61/article.pdf."""
import numpy as np

def div(arr):
    """Computes the discrete divergence of a vector array."""
    out = np.zeros_like(arr)
    out[0, -1, :, ...] = -1 * arr[0, -2, :, ...]
    out[0, 1:-1, :, ...] = arr[0, 1:-1, :, ...] - arr[0, :-2, :, ...]
    out[1, :, -1, ...] = -1 * arr[1, :, -2, ...]
    out[1, :, 1:-1, ...] = arr[1, :, 1:-1, ...] - arr[1, :, :-2, ...]
    return out.sum(axis=0)


def magnitude(arr, axis=0, keepdims=False):
    """Computes the element-wise magnitude of a vector array."""
    return np.linalg.norm(arr, ord=2, axis=axis, keepdims=keepdims)


def grad(arr):
    """Computes the discrete gradient of an image."""
    out = np.zeros((2,) + arr.shape, arr.dtype)
    out[0, :-1, :, ...] = arr[1:, :, ...] - arr[:-1, :, ...]
    out[1, :, :-1, ...] = arr[:, 1:, ...] - arr[:, :-1, ...]
    return out


def tv_denoise_chambolle(image, strength, step_size=0.2, eps=3e-3):
    """Total variation image denoising with Chambolle's projection algorithm."""
    image = np.atleast_3d(image)
    p = np.zeros((2,) + image.shape, image.dtype)
    normalized_img = image / strength
    diff = np.inf
    while diff > eps:
        grad_div_p_i = grad(div(p) - normalized_img)
        magnitude_denoise = magnitude(grad_div_p_i, axis=(0, -1), keepdims=True)
        new_p = (p + step_size * grad_div_p_i) / (1 + step_size * magnitude_denoise)
        diff = np.max(magnitude(new_p - p))
        p[:] = new_p

    return np.squeeze(image - strength * div(p))