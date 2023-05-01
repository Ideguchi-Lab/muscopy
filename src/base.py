try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False


def convert_to_png(array, nonzero_range=None):
    # set elements to 0 outside the specified range.
    array_extracted = array.copy()
    if nonzero_range is not None:
        array_extracted[array_extracted < nonzero_range[0]] = nonzero_range[0]
        array_extracted[array_extracted > nonzero_range[1]] = nonzero_range[1]
    dr = xp.max(array_extracted) - xp.min(array_extracted)
    origin_shift = xp.min(array_extracted)
    # shift
    array_converted = array_extracted - origin_shift
    x = (2**16 - 1) / dr
    assert x >= 0
    array_converted = array_converted * x
    array_converted = array_converted.astype(xp.uint16)
    return array_converted
