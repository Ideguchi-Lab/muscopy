import muscopy.cfg as mcfg
from muscopy import QPI, QPIParameters

if mcfg._cp:
    import cupy as xp
else:
    import numpy as xp


def decode_adimec(array):
    array_upper = array[::2, :]
    array_buttom = array[1::2, :]
    array_buttom = xp.flip(array_buttom, 0)
    return xp.concatenate((array_upper, array_buttom))


def qpi_converter(array, array_ref):
    if mcfg._cp:
        array = xp.array(array)
        array_ref = xp.array(array_ref)
    array = decode_adimec(array)[:1439, :1439]
    array_ref = decode_adimec(array_ref)[:1439, :1439]

    params = QPIParameters(
        wavelength=532 * 10 ** (-9),
        NA=0.6,
        img_shape=(1439, 1439),
        pixelsize=12 * 10 ** (-6) / 40 / 4,
        offaxis_center=(1238, 716),
    )
    result = QPI(array, array_ref, params)

    result = result[1:-2, 1:-2]

    if mcfg._cp:
        result = xp.asnumpy(result)
    return [result]
