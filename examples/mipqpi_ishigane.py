try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False

from microscopy_converters.muscopy.mipqpi import QPIParameters, convert_to_png, decode_adimec, mipqpi


def mipqpi_converter_adimec(array1, array2, cutoff):
    if _cp:
        array1 = xp.array(array1)
        array2 = xp.array(array2)
    image_center = (719, 719)
    array1 = decode_adimec(array1)[:1439, :1439]
    array2 = decode_adimec(array2)[:1439, :1439]
    params = QPIParameters(
        wavelength=532 * 10 ** (-9),
        NA=0.6,
        img_shape=(1439, 1439),
        img_center=image_center,
        pixelsize=12 * 10 ** (-6) / 40 / 4,
        center=(1238, 716),
    )
    params.calc_params()
    result = mipqpi(array1, array2, params)

    result1 = convert_to_png(result, [0, cutoff])
    result2 = convert_to_png(result, [-cutoff, 0])
    if _cp:
        result1 = xp.asnumpy(result1)
        result2 = xp.asnumpy(result2)

    return [result1, result2]
