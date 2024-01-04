# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import typing

sys.path.append("../..")
from src.odt import ODTParameters
from src.qpi import make_disk

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False


# %%


def generate_3D_sphere(
    shape: tuple | int, radius: float, center: tuple | None = None
) -> xp.ndarray:
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

    return array_2d_fft.T


def extract3Dto2D_minimum(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    oblique_shift: tuple[int, int],
) -> xp.ndarray:
    array_2d_fft = xp.zeros(
        (params.aperturesize, params.aperturesize), dtype=xp.complex128
    )

    xx, yy = xp.meshgrid(
        xp.arange(2 * params.aperturesize + 1),
        xp.arange(2 * params.aperturesize + 1),
        indexing="ij",
    )
    circle = (xx - params.aperturesize // 2) ** 2 + (yy - params.aperturesize // 2) ** 2
    circle = circle < (params.aperturesize // 2) ** 2

    for i, j in zip(*xp.where(circle)):
        Kz = int(
            xp.sqrt(
                params.ki_mag**2
                - (i - params.aperturesize // 2) ** 2
                - (j - params.aperturesize // 2) ** 2
            )
            - xp.sqrt(
                params.ki_mag**2 - oblique_shift[0] ** 2 - oblique_shift[1] ** 2
            )
        )
        # print(Kz)
        # array_2d_fft[i, j] = array_3d_fft[
        #     i
        #     - params.aperturesize // 2
        #     - oblique_shift[0]
        #     + array_3d_fft.shape[0] // 2,
        #     j
        #     - params.aperturesize // 2
        #     - oblique_shift[1]
        #     + array_3d_fft.shape[1] // 2,
        #     Kz + array_3d_fft.shape[2] // 2,
        # ]
        array_2d_fft[i, j] = array_3d_fft[
            i
            - params.aperturesize // 2
            - oblique_shift[0]
            + array_3d_fft.shape[0] // 2,
            j
            - params.aperturesize // 2
            - oblique_shift[1]
            + array_3d_fft.shape[1] // 2,
            Kz + array_3d_fft.shape[2] // 2,
        ]

    return array_2d_fft.T
    # return array_2d_fft


# def map_2d_to_3d(array_fft_2d: xp.ndarray, array_fft_3d: xp.ndarray, params: ODTParameters, oblique_center: tuple[int, int])->xp.ndarray:
#     xx, yy = xp.meshgrid(xp.arange(params.img_shape[0]), xp.arange(params.img_shape[1]), indexing="ij")
#     circle = (xx - params.offaxis_center[0]) ** 2 + (yy - params.offaxis_center[1]) ** 2
#     circle = circle < (params.aperturesize // 2) ** 2

#     for i, j in zip(*xp.where(circle)):
#         Kz = int(xp.sqrt(params.ki_mag**2 - (i - params.offaxis_center[0]) ** 2 - (j - params.offaxis_center[1]) ** 2) - xp.sqrt(params.ki_mag**2 - ))


def generate_test_data(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    illumi_angle_step=30,
    NA_illumi=1.0,
    path: str = "odt_test_data",
    approx="Rytov",
):
    assert approx in ["Rytov", "Born"]
    num = int(360 / illumi_angle_step)
    E_initial = xp.ones(
        (2 * params.aperturesize + 1, 2 * params.aperturesize + 1), dtype=xp.complex128
    )
    # include noise
    E_initial = (
        E_initial
        + xp.random.normal(0, 0.01, E_initial.shape)
        + xp.random.normal(0, 0.01, E_initial.shape) * 1j
    )
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
    os.mkdir(path)

    ki = round(NA_illumi / params.wav / params.freq_per_pixel) + 1
    for i in range(num):
        oblique_shift = (
            int(ki * xp.cos(illumi_angle_step * i / 360 * 2 * xp.pi)),
            int(ki * xp.sin(illumi_angle_step * i / 360 * 2 * xp.pi)),
        )
        test_data_fft_cropped = extract3Dto2D_minimum(
            array_3d_fft, params, oblique_shift=oblique_shift
        )
        fft_extent = xp.zeros(
            (2 * params.aperturesize + 1, 2 * params.aperturesize + 1),
            dtype=xp.complex128,
        )
        fft_extent[
            params.aperturesize
            - oblique_shift[1]
            - params.aperturesize // 2 : params.aperturesize
            - oblique_shift[1]
            + params.aperturesize // 2
            + 1,
            params.aperturesize
            - oblique_shift[0]
            - params.aperturesize // 2 : params.aperturesize
            - oblique_shift[0]
            + params.aperturesize // 2
            + 1,
        ] = test_data_fft_cropped
        test_data_extent = xp.fft.ifft2(xp.fft.ifftshift(fft_extent))
        test_data_extent[:1, :] = 0
        test_data_extent[:, :1] = 0
        if approx == "Rytov":
            E_test = E_initial * xp.exp(test_data_extent / E_initial)
        elif approx == "Born":
            E_test = E_initial + test_data_extent
        E_test_fft = xp.fft.fftshift(xp.fft.fft2(E_test))
        low_pass = make_disk(
            (
                params.aperturesize - oblique_shift[0],
                params.aperturesize - oblique_shift[1],
            ),
            params.aperturesize // 2,
            E_initial.shape,
        )
        E_test_fft = E_test_fft * low_pass
        test_data_fft = xp.zeros(params.img_shape, dtype=xp.complex128)
        test_data_fft[
            params.offaxis_center[0]
            - params.aperturesize // 2 : params.offaxis_center[0]
            + params.aperturesize // 2
            + 1,
            params.offaxis_center[1]
            - params.aperturesize // 2 : params.offaxis_center[1]
            + params.aperturesize // 2
            + 1,
        ] = E_test_fft[
            params.aperturesize
            - oblique_shift[1]
            - params.aperturesize // 2 : params.aperturesize
            - oblique_shift[1]
            + params.aperturesize // 2
            + 1,
            params.aperturesize
            - oblique_shift[0]
            - params.aperturesize // 2 : params.aperturesize
            - oblique_shift[0]
            + params.aperturesize // 2
            + 1,
        ]

        test_data_fft = test_data_fft.T

        test_data = xp.fft.ifft2(xp.fft.ifftshift(test_data_fft))
        test_data[:2, :] = 0
        test_data[:, :2] = 0

        # save
        xp.save(f"{path}/{int(illumi_angle_step * i):03}.npy", test_data)


# %%
