# %%
import matplotlib.pyplot as plt
import numpy as np
from ilabvis import CursorVisualizer, SlicingVisualizer

# %%

xx, yy = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
circle = (xx - 50) ** 2 + (yy - 50) ** 2 < 25**2

cursorvis = CursorVisualizer(circle)
cursorvis.run()

# %%
circle_roll = np.roll(circle, circle.shape[0] // 2, axis=0)
cursorvis = CursorVisualizer(circle_roll)
cursorvis.run()

# %%
circle_fft = np.fft.fftshift(np.fft.fft2(circle))
circle_fft_abs = np.abs(circle_fft)
cursorvis = CursorVisualizer(circle_fft_abs)
cursorvis.run()

# %%
# roll half of the array shape along x axis
circle_fft_roll = np.roll(circle_fft, circle_fft.shape[0] // 2, axis=0)
cursorvis = CursorVisualizer(np.abs(circle_fft_roll))
cursorvis.run()

# %%
# ifft
circle_roll = np.fft.ifft2(circle_fft_roll)

cursorvis = CursorVisualizer(np.abs(circle_roll))
cursorvis.run()
# %%

# test fftshift

test = (np.arange(10**3)).reshape(10, 10, 10)
test = test**2
slicevis = SlicingVisualizer(test)
slicevis.run()

# %%
xx, yy, zz = np.meshgrid(np.arange(100), np.arange(100), np.arange(100), indexing="ij")

sphere = (xx - 50) ** 2 + (yy - 50) ** 2 + (zz - 50) ** 2 < 10**2

slicevis = SlicingVisualizer(sphere)
slicevis.run()

# %%
sphere_fft = np.fft.fftshift(np.fft.fftn(sphere))
slicevis = SlicingVisualizer(np.abs(sphere_fft))
slicevis.run()

# %%
# transpose xy
sphere_fft = np.transpose(sphere_fft, axes=(1, 0, 2))

slicevis = SlicingVisualizer(np.abs(sphere_fft))
slicevis.run()

# %%
sphere_recovered = np.fft.ifftn(np.fft.ifftshift(sphere_fft))
slicevis = SlicingVisualizer(np.real(sphere_recovered))
slicevis.run()

# %%
