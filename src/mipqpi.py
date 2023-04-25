try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False


class QPIParameters:
    def __init__(self, wavelength, NA, img_shape, img_center, pixelsize, center):
        self.wav = wavelength
        self.NA = NA
        self.img_shape = img_shape
        self.img_center = img_center
        self.pixelsize = pixelsize
        self.center = center

    def calc_params(self):
        self.dim = self.img_shape[0]
        self.freq_per_pixel = 1 / (self.pixelsize * self.dim)
        self.aperturesize = 2 * round(2 * self.NA / self.wav / self.freq_per_pixel / 2) + 1


def mipqpi(array_on, array_off, params, print_backend=False):
    assert array_on.shape == array_off.shape
    array_off_fft = xp.fft.fftshift(xp.fft.fft2(array_off))
    mask = make_circle(params.center, params.aperturesize, params.img_shape)
    array_off_fft = array_off_fft * mask
    array_off_fft = array_off_fft[
        params.center[1] - params.aperturesize // 2 : params.center[1] + params.aperturesize // 2 + 1,
        params.center[0] - params.aperturesize // 2 : params.center[0] + params.aperturesize // 2 + 1,
    ]
    array_off = xp.fft.ifft2(xp.fft.ifftshift(array_off_fft))

    array_on_fft = xp.fft.fftshift(xp.fft.fft2(array_on))
    array_on_fft = array_on_fft * mask
    array_on_fft = array_on_fft[
        params.center[1] - params.aperturesize // 2 : params.center[1] + params.aperturesize // 2 + 1,
        params.center[0] - params.aperturesize // 2 : params.center[0] + params.aperturesize // 2 + 1,
    ]
    array_on = xp.fft.ifft2(xp.fft.ifftshift(array_on_fft))

    dif_phase = xp.angle(array_on / array_off)

    if print_backend:
        backend = "cupy" if _cp else "numpy"
        print(backend + " is used as a backend")

    return dif_phase


def make_circle(center, radius, array_shape, highpass=False):
    """internal method. return circle filled with 1.

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
    xx, yy = xp.meshgrid(xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing="xy")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def decode_adimec(array):
    array_upper = array[::2, :]
    array_buttom = array[1::2, :]
    array_buttom = xp.flip(array_buttom, 0)
    return xp.concatenate((array_upper, array_buttom))


def convert_to_png(array, nonzero_range=None):
    # set elements to 0 outside the specified range.
    array_extracted = array.copy()
    if range is not None:
        array_extracted[array_extracted < nonzero_range[0]] = 0
        array_extracted[array_extracted > nonzero_range[1]] = 0
    dr = xp.max(array_extracted) - xp.min(array_extracted)
    origin_shift = xp.min(array_extracted)
    # shift
    array_converted = array_extracted - origin_shift
    x = 2**16 / dr
    assert x >= 0
    array_converted = array_converted * x
    array_converted = array_converted.astype(xp.uint16)
    return array_converted
