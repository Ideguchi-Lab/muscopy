# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append("../..")
from src.odt import ODTParameters

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False


# %%


def generate_3D_sphere(shape, radius, center=None):
    """Generate 3D sphere. The sphere is filled with 1, otherwise 0.

    Args:
        shape (tuple): shape of the tensor
        radius (float): radius of the sphere
        center (tuple, optional): center of the sphere. Defaults to None.

    Returns:
        xp.ndarray: 3D sphere
    """
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    if center is None:
        center = tuple([int(i / 2) for i in shape])
    xx, yy, zz = xp.meshgrid(
        xp.arange(shape[0]), xp.arange(shape[1]), xp.arange(shape[2]), indexing="ij"
    )
    sphere = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    sphere = sphere < radius**2
    return sphere


def extract3Dto2D(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    illumi_angle=0,
    NA_illumi=1.0,
) -> xp.ndarray:
    array_2d_fft = xp.zeros(params.img_shape)
    ki = round(NA_illumi / params.wav / params.freq_per_pixel) + 1
    oblique_shift = (
        int(ki * xp.cos(illumi_angle / 360 * 2 * xp.pi)),
        int(ki * xp.sin(illumi_angle / 360 * 2 * xp.pi)),
    )

    xx, yy = xp.meshgrid(
        xp.arange(params.img_shape[0]), xp.arange(params.img_shape[1]), indexing="ij"
    )
    circle = (xx - params.offaxis_center[0]) ** 2 + (yy - params.offaxis_center[1]) ** 2
    circle = circle < (params.aperturesize // 2) ** 2

    for i, j in zip(*xp.where(circle)):
        Kz = int(
            xp.sqrt(
                params.ki_mag**2
                - (i - params.offaxis_center[0]) ** 2
                - (j - params.offaxis_center[1]) ** 2
            )
            - xp.sqrt(
                params.ki_mag**2 - oblique_shift[0] ** 2 - oblique_shift[1] ** 2
            )
        )
        array_2d_fft[i, j] = array_3d_fft[
            i
            - params.offaxis_center[0]
            - oblique_shift[0]
            + array_3d_fft.shape[0] // 2,
            j
            - params.offaxis_center[1]
            - oblique_shift[1]
            + array_3d_fft.shape[1] // 2,
            Kz + array_3d_fft.shape[2] // 2,
        ]

    return array_2d_fft


def generate_test_data(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    illumi_angle_step=30,
    NA_illumi=1.0,
):
    num = int(360 / illumi_angle_step)
    if os.path.exists("odt_test_data"):
        import shutil

        shutil.rmtree("odt_test_data")
    os.mkdir("odt_test_data")
    for i in range(num):
        test_data_fft = extract3Dto2D(
            array_3d_fft, params, illumi_angle_step * i, NA_illumi
        )
        test_data = xp.fft.ifft2(xp.fft.ifftshift(test_data_fft))
        if _cp:
            test_data = xp.asnumpy(test_data)
        # save
        np.save(f"odt_test_data/{int(illumi_angle_step * i):03}.npy", test_data)


# %%
