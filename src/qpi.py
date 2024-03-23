try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False


class QPIParameters:
    def __init__(
        self, wavelength, NA, img_shape, img_center, pixelsize, offaxis_center
    ):
        self.wav = wavelength
        self.NA = NA
        self.img_shape = img_shape
        self.img_center = img_center
        self.pixelsize = pixelsize
        self.offaxis_center = offaxis_center

        self.calc_params()

    def calc_params(self):
        self.dim = self.img_shape[0]
        self.freq_per_pixel = 1 / (self.pixelsize * self.dim)  # 1 / L
        self.aperturesize = (
            2 * round(self.NA / self.wav / self.freq_per_pixel) + 1
        )  # 2 * f_BW + 1

    def print_all_parameters(self):
        print(f"{self.dim=}")
        print(f"{self.freq_per_pixel=}")
        print(f"{self.aperturesize=}")
        print(f"{self.offaxis_center=}")
        print(f"{self.img_center=}")
        print(f"{self.pixelsize=}")
        print(f"{self.img_shape=}")
        print(f"{self.NA=}")
        print(f"{self.wav=}")


def make_disk(center, radius, array_shape, highpass=False):
    """internal method. return disk filled with 1.

    Args:
        center (tuple of int): center position of the circle.
        radius (float): radius of the circle
        shape (_type_): _description_
        highpass (bool, optional): _description_. Defaults to False.

    Returns:
        xp.array: array whose pass area is filled with 1, otherwise 0.
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = xp.meshgrid(
        xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing="ij"
    )
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def qpi(array, reference, params):
    assert array.shape == reference.shape
    reference_fft = xp.fft.fftshift(xp.fft.fft2(reference))
    mask = make_disk(params.off_axis, params.aperturesize / 2, params.img_shape)
    reference_fft = reference_fft * mask
    reference_fft = reference_fft[
        params.off_axis[0]
        - params.aperturesize // 2 : params.off_axis[0]
        + params.aperturesize // 2
        + 1,
        params.off_axis[1]
        - params.aperturesize // 2 : params.off_axis[1]
        + params.aperturesize // 2
        + 1,
    ]
    reference = xp.fft.ifft2(xp.fft.ifftshift(reference_fft))

    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    array_fft = array_fft * mask
    array_fft = array_fft[
        params.off_axis[1]
        - params.aperturesize // 2 : params.off_axis[0]
        + params.aperturesize // 2
        + 1,
        params.off_axis[0]
        - params.aperturesize // 2 : params.off_axis[1]
        + params.aperturesize // 2
        + 1,
    ]
    array = xp.fft.ifft2(xp.fft.ifftshift(array_fft))

    # mean_phase = xp.mean(xp.angle(reference))

    dif_phase = xp.angle(array / reference)

    # dif_phase = dif_phase - mean_phase

    return dif_phase


def mipqpi(array_on, array_off, params, print_backend=False):
    assert array_on.shape == array_off.shape
    array_off_fft = xp.fft.fftshift(xp.fft.fft2(array_off))
    mask = make_disk(params.offaxis_center, params.aperturesize, params.img_shape)
    array_off_fft = array_off_fft * mask
    array_off_fft = array_off_fft[
        params.offaxis_center[0]
        - params.aperturesize // 2 : params.offaxis_center[0]
        + params.aperturesize // 2
        + 1,
        params.offaxis_center[1]
        - params.aperturesize // 2 : params.offaxis_center[1]
        + params.aperturesize // 2
        + 1,
    ]
    array_off = xp.fft.ifft2(xp.fft.ifftshift(array_off_fft))

    array_on_fft = xp.fft.fftshift(xp.fft.fft2(array_on))
    array_on_fft = array_on_fft * mask
    array_on_fft = array_on_fft[
        params.offaxis_center[0]
        - params.aperturesize // 2 : params.offaxis_center[0]
        + params.aperturesize // 2
        + 1,
        params.offaxis_center[1]
        - params.aperturesize // 2 : params.offaxis_center[1]
        + params.aperturesize // 2
        + 1,
    ]
    array_on = xp.fft.ifft2(xp.fft.ifftshift(array_on_fft))

    dif_phase = xp.angle(array_on / array_off)

    if print_backend:
        backend = "cupy" if _cp else "numpy"
        print(backend + " is used as a backend")

    return dif_phase
