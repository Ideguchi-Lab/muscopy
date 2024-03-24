try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False


def fft_log(array):
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    array = xp.log(xp.abs(array_fft))
    return array
