try:
    import cupy as xp
    _cp = True
except:
    import numpy as xp
    _cp = False


class QPIParameters:
    def __init__(self, wavelength, NA, img_size, pixelsize, center):
        self.wav = wavelength
        self.NA = NA
        self.img_size = img_size
        self.pixelsize = pixelsize
        self.center = center

    def calc_params(self):
        self.dim = self.img_size
        self.freq_per_pixel = 1/(self.pixelsize * self.dim)
        self.aperturesize = 2 * \
            round(2*self.NA/self.wav/self.freq_per_pixel/2) + 1


def mipqpi(array_on, array_off, print_backend=False):
    assert array_on.shape == array_off.shape
    if _cp:
        array_on = xp.array(array_on)
        array_off = xp.array(array_off)
    img_shape = array_on.shape
    img_center = (img_shape[0]//2, img_shape[1]//2)
    params = QPIParameters(wavelength=532*10**(-9), NA=0.6,
                           img_size=img_shape[0], pixelsize=12*10**(-6)/40/4, center=(1239, 717))

    params.calc_params()
    array_off_fft = xp.fft.fftshift(xp.fft.fft2(array_off))
    mask = make_circle(params.center, params.aperturesize, img_shape)
    array_off_fft = array_off_fft * mask
    array_off_fft = array_off_fft[params.center[1]-params.aperturesize//2:params.center[1] +
                                  params.aperturesize//2 + 1, params.center[0]-params.aperturesize//2:params.center[0]+params.aperturesize//2 + 1]
    array_off = xp.fft.ifft2(xp.fft.ifftshift(array_off_fft))

    array_on_fft = xp.fft.fftshift(xp.fft.fft2(array_on))
    array_on_fft = array_on_fft * mask
    array_on_fft = array_on_fft[params.center[1]-params.aperturesize//2:params.center[1] +
                                params.aperturesize//2 + 1, params.center[0]-params.aperturesize//2:params.center[0]+params.aperturesize//2 + 1]
    array_on = xp.fft.ifft2(xp.fft.ifftshift(array_on_fft))

    dif_phase = xp.angle(array_on / array_off)

    if _cp:
        dif_phase = xp.asnumpy(dif_phase)

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
    xx, yy = xp.meshgrid(
        xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing='xy')
    circle = (xx-center[0])**2 + (yy-center[1])**2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def convert_to_png(array, range=None):
    # set elements to 0 outside the specified range.
    if range is not None:
        array[array < range[0]] = 0
        array[array > range[1]] = 0
    dr = xp.max(array) - xp.min(array)
    origin_shift = xp.min(array)
    # shift
    array_converted = array - origin_shift
    x = (2**16 / dr)
    assert x >= 0
    array_converted = array_converted * x
    array_converted = array_converted.astype(xp.uint16)
    return array_converted
