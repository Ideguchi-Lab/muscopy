import muscopy.cfg as mcfg
from muscopy import QPIParameters, MIPQPI

if mcfg._cp:
    import cupy as xp
else:
    import numpy as xp


def decode_adimec(array):
    array_upper = array[::2, :]
    array_buttom = array[1::2, :]
    array_buttom = xp.flip(array_buttom, 0)
    return xp.concatenate((array_upper, array_buttom))


def mipqpi_converter_adimec(array1, array2):
    array1 = decode_adimec(array1)[:1439, :1439]
    array2 = decode_adimec(array2)[:1439, :1439]
    params = QPIParameters(
        wavelength=532 * 10 ** (-9),
        NA=0.6,
        img_shape=(1439, 1439),
        pixelsize=12 * 10 ** (-6) / 40 / 4,
        offaxis_center=(1238, 716),
    )
    result1 = MIPQPI(array1, array2, params)
    result2 = -result1

    if mcfg._cp:
        result1 = xp.asnumpy(result1)
        result2 = xp.asnumpy(result2)

    return [result1, result2]
