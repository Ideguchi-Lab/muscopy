import muscopy.cfg as mcfg

if mcfg._cp:
    import cupy as xp
else:
    import numpy as xp


def fft_log(array):
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    array = xp.log(xp.abs(array_fft))
    return array
