from __future__ import annotations

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False


class QPIParameters:
    def __init__(
        self,
        wavelength: float,
        NA: float,
        img_shape: tuple[int, int],
        pixelsize: float,
        offaxis_center: tuple[int, int],
    ):
        """QPIParameters class for QPI calculation

        Args:
            wavelength (float): wavelength of the probe
            NA (float): Numerical aperture of the objective
            img_shape (tuple[int, int]): shape of the image
            pixelsize (float): image pixel size
            offaxis_center (tuple[int, int]): center position of the off axis holography in the Fourier domain
        """
        self.wav = wavelength
        self.NA = NA
        self.img_shape = img_shape
        self.pixelsize = pixelsize
        self.offaxis_center = offaxis_center

        self._calc_params()

    def _calc_params(self):
        """Calculate parameters for QPI calculation. This method is called in __init__ method."""

        self.img_center = (self.img_shape[0] // 2, self.img_shape[1] // 2)
        self.dim = self.img_shape[0]
        self.freq_per_pixel = 1 / (self.pixelsize * self.dim)  # 1 / L
        self.aperturesize = 2 * round(self.NA / self.wav / self.freq_per_pixel) + 1  # 2 * f_BW + 1

    def print_all_parameters(self):
        """Print all parameters in the QPIParameters class"""
        for key, value in vars(self).items():
            print(f"{key}={value}")


def make_disk(center: tuple[int, int], radius: float, array_shape: tuple[int, int], highpass: bool = False) -> xp.array:
    """make disk mask for filtering

    Args:
        center (tuple[int, int]): center position of the disk mask
        radius (float): radius of the disk mask
        array_shape (tuple[int, int]): shape of the array
        highpass (bool, optional): Filter low frequency or not. Defaults to False.

    Returns:
        xp.array: disk mask
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = xp.meshgrid(xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def crop_array(array: xp.array, center: tuple[int, int], width: int) -> xp.array:
    """internal method. crop the array with specified center and width.

    Args:
        array (xp.array): array to be cropped
        center (tuple of int): center position of the cropped array
        width (int): width of the cropped array

    Returns:
        xp.array: cropped array
    """
    return array[
        center[0] - width // 2 : center[0] + width // 2 + 1,
        center[1] - width // 2 : center[1] + width // 2 + 1,
    ]


def get_field(array: xp.array, params: QPIParameters, crop_center: bool = False, c_r: int = 5) -> xp.array:
    """internal method. get the electric field from the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: field of the array
    """
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    mask = make_disk(params.offaxis_center, params.aperturesize / 2, params.img_shape)
    array_fft = array_fft * mask

    if crop_center:
        mask_highpass = make_disk(params.offaxis_center, c_r, params.img_shape, highpass=True)
        array_fft = array_fft * mask_highpass

    array_fft = crop_array(array_fft, params.offaxis_center, params.aperturesize)
    array = xp.fft.ifft2(xp.fft.ifftshift(array_fft))
    return array


def qpi(
    array: xp.array,
    reference: xp.array,
    params: QPIParameters,
    phase_offset_regs: list[tuple[tuple[int, int], tuple[int, int]]] = None,
) -> xp.array:
    """Quantitative phase imaging (QPI) calculation

    Args:
        array (xp.array): on-axis hologram
        reference (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class
        phase_offset_regs (list[tuple[tuple[int, int], tuple[int, int]]], optional): regions for phase offset calculation. Defaults to None.

    Returns:
        xp.array: QPI phase image
    """
    assert array.shape == reference.shape

    array_field = get_field(array, params)
    ref_array_field = get_field(reference, params)

    dif_phase = xp.angle(array_field / ref_array_field)

    # remove phase offset
    if phase_offset_regs is not None:
        phase_offset_ls = []
        for reg in phase_offset_regs:
            phase_offset_ls.append(xp.mean(dif_phase[reg[0][0] : reg[0][1], reg[1][0] : reg[1][1]]))
        phase_offset = xp.mean(phase_offset_ls)
        dif_phase -= phase_offset

    return dif_phase


def mipqpi(
    array_on: xp.array,
    array_off: xp.array,
    params: QPIParameters,
    phase_offset_regs: list[tuple[tuple[int, int], tuple[int, int]]] = None,
    crop_center: bool = False,
) -> xp.array:
    """Mid-infrared photothermal quantitative phase imaging (MIP-QPI) calculation

    Args:
        array_on (xp.array): on-axis hologram
        array_off (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class
        phase_offset_regs (list[tuple[tuple[int, int], tuple[int, int]]], optional): regions for phase offset calculation. Defaults to None.
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: MIP-QPI phase image
    """
    assert array_on.shape == array_off.shape
    array_on_field = get_field(array_on, params, crop_center=crop_center)
    array_off_field = get_field(array_off, params, crop_center=crop_center)

    dif_phase = xp.angle(array_on_field / array_off_field)

    # remove phase offset
    if phase_offset_regs is not None:
        phase_offset_ls = []
        for reg in phase_offset_regs:
            phase_offset_ls.append(xp.mean(dif_phase[reg[0][0] : reg[0][1], reg[1][0] : reg[1][1]]))
        phase_offset = xp.mean(phase_offset_ls)
        dif_phase -= phase_offset

    return dif_phase
