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
        self.aperturesize = (
            2 * round(2 * self.NA / self.wav / self.freq_per_pixel / 2) + 1
        )



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
        xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing="xy"
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
    mask = make_disk(params.center, params.aperturesize/2, params.img_shape)
    reference_fft = reference_fft * mask
    reference_fft = reference_fft[
        params.center[1]
        - params.aperturesize // 2: params.center[1]
        + params.aperturesize // 2
        + 1,
        params.center[0]
        - params.aperturesize // 2: params.center[0]
        + params.aperturesize // 2
        + 1,
    ]
    reference = xp.fft.ifft2(xp.fft.ifftshift(reference_fft))

    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    array_fft = array_fft * mask
    array_fft = array_fft[
        params.center[1]
        - params.aperturesize // 2: params.center[1]
        + params.aperturesize // 2
        + 1,
        params.center[0]
        - params.aperturesize // 2: params.center[0]
        + params.aperturesize // 2
        + 1,
    ]
    array = xp.fft.ifft2(xp.fft.ifftshift(array_fft))

    # mean_phase = xp.mean(xp.angle(reference))

    dif_phase = xp.angle(array / reference)

    # dif_phase = dif_phase - mean_phase

    return dif_phase

