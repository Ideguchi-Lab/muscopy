try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False

from qpi import QPIParameters, convert_to_png, decode_adimec, qpi


def qpi_converter(array, array_ref):
    if _cp:
        array = xp.array(array)
        array_ref = xp.array(array_ref)
    image_center = (719, 719)
    array = decode_adimec(array)[:1439, :1439]
    array_ref = decode_adimec(array_ref)[:1439, :1439]

    params = QPIParameters(
        wavelength=532 * 10 ** (-9),
        NA=0.6,
        img_shape=(1439, 1439),
        img_center=image_center,
        pixelsize=12 * 10 ** (-6) / 40 / 4,
        center=(1238, 716),
    )
    params.calc_params()
    result = qpi(array, array_ref, params)
    result = convert_to_png(result, [0, 0.05])

    result = result[1:-2, 1:-2]

    if _cp:
        result = xp.asnumpy(result)
    return [result]
