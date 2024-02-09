# %%
import numpy as np
import os
import sys

sys.path.append("../..")
from src.odt import ODTParameters, find_max_args
from src.qpi import make_disk

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

EDGE_SIZE = 0

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


def generate_3D_slope(shape: tuple, axis: str) -> xp.ndarray:
    xx, yy, zz = xp.meshgrid(
        xp.arange(shape[0]), xp.arange(shape[1]), xp.arange(shape[2]), indexing="ij"
    )
    if axis == "x":
        slope = xx
    elif axis == "y":
        slope = yy
    elif axis == "z":
        slope = zz

    return slope


def extract3Dto2D_minimum(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    oblique_shift: tuple[int, int],
    return_index_map=False,
) -> xp.ndarray:
    xx, yy = xp.meshgrid(
        xp.arange(array_3d_fft.shape[0]),
        xp.arange(array_3d_fft.shape[1]),
        indexing="ij",
    )
    _, _, zz = xp.meshgrid(
        xp.arange(array_3d_fft.shape[0]),
        xp.arange(array_3d_fft.shape[1]),
        xp.arange(array_3d_fft.shape[2]),
        indexing="ij",
    )
    circle = (xx - params.aperturesize - oblique_shift[0]) ** 2 + (
        yy - params.aperturesize - oblique_shift[1]
    ) ** 2
    circle = circle < (params.aperturesize // 2) ** 2
    Kz_circle = xp.sqrt(
        params.ki_mag**2
        - (xx - params.aperturesize - oblique_shift[0]) ** 2
        - (yy - params.aperturesize - oblique_shift[1]) ** 2
    ) - xp.sqrt(params.ki_mag**2 - oblique_shift[0] ** 2 - oblique_shift[1] ** 2)

    Kz_value = (Kz_circle + array_3d_fft.shape[2] // 2) * circle
    Kz_tile = xp.tile(Kz_value, (array_3d_fft.shape[2], 1, 1))
    Kz_tile = Kz_tile.transpose(1, 2, 0)

    Kz_tile = Kz_tile.astype(xp.int64)

    Kz_tile -= Kz_tile == 0  # to avoid 0 index match with zz

    Kz_index = zz == Kz_tile

    if return_index_map:
        return Kz_index

    array_cropped = array_3d_fft * Kz_index

    array_2d_fft = xp.sum(array_cropped, axis=2)

    return array_2d_fft


def generate_test_data(
    array_3d_fft: xp.ndarray,
    params: ODTParameters,
    illumi_angle_step=30,
    NA_illumi=1.0,
    path: str = "odt_test_data",
    approx="Rytov",
    if_save=False,
):
    assert approx in ["Rytov", "Born"]
    num = int(360 / illumi_angle_step)
    E_initial = xp.ones(
        (2 * params.aperturesize + 1, 2 * params.aperturesize + 1), dtype=xp.complex128
    )
    # include noise
    # E_initial = (
    #     E_initial
    #     + xp.random.normal(0, 0.01, E_initial.shape)
    #     + xp.random.normal(0, 0.01, E_initial.shape) * 1j
    # )
    if os.path.exists(path):
        import shutil

        shutil.rmtree(path)
    os.mkdir(path)

    ki = round(NA_illumi / params.wav / params.freq_per_pixel)  # + 1 why +1?
    for i in range(num):
        oblique_shift = (
            int(ki * xp.cos(illumi_angle_step * i / 360 * 2 * xp.pi)),
            int(ki * xp.sin(illumi_angle_step * i / 360 * 2 * xp.pi)),
        )
        test_data_fft_cropped = extract3Dto2D_minimum(
            array_3d_fft, params, oblique_shift=oblique_shift
        )
        fft_extent = test_data_fft_cropped
        norm_fft_extent = fft_extent * params.k_unit
        norm_test_data_extent = xp.fft.ifft2(
            xp.fft.ifftshift(norm_fft_extent), norm="ortho"
        )
        test_data_extent = norm_test_data_extent / params.imgpx_unit
        # test_data_extent[:EDGE_SIZE, :] = 1e-6
        # test_data_extent[:, :EDGE_SIZE] = 1e-6
        if approx == "Rytov":
            E_test = E_initial * xp.exp(test_data_extent)
        elif approx == "Born":
            E_test = E_initial + E_initial * test_data_extent
        if if_save:
            xp.save("./test_data_extent.npy", test_data_extent)
        norm_E_test = E_test * params.imgpx_unit
        norm_E_test_fft = xp.fft.fftshift(xp.fft.fft2(norm_E_test, norm="ortho"))
        E_test_fft = norm_E_test_fft / params.k_unit
        low_pass = make_disk(
            (
                params.aperturesize + oblique_shift[0],
                params.aperturesize + oblique_shift[1],
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
            + oblique_shift[0]
            - params.aperturesize // 2 : params.aperturesize
            + oblique_shift[0]
            + params.aperturesize // 2
            + 1,
            params.aperturesize
            + oblique_shift[1]
            - params.aperturesize // 2 : params.aperturesize
            + oblique_shift[1]
            + params.aperturesize // 2
            + 1,
        ]

        xp.save(f"{path}/{int(illumi_angle_step * i):03}.npy", test_data_fft)

        # test_data = xp.fft.ifft2(xp.fft.ifftshift(test_data_fft))
        # test_data[:2, :] = 0
        # test_data[:, :2] = 0

        # save
        # xp.save(f"{path}/{int(illumi_angle_step * i):03}.npy", test_data)
