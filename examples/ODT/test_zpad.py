# %%
try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

from ilabvis import SlicingVisualizer

import muscopy as mus

# %%
rand_mat = xp.random.rand(100, 100, 100)
# xx, yy, zz = xp.meshgrid(xp.arange(100), xp.arange(100), xp.arange(100), indexing="ij")
# rand_mat = ((xx - 50) ** 2 + (yy - 50) ** 2 + (zz - 50) ** 2 < 10**2).astype(xp.float32)
rand_mat = rand_mat * 10
print("abs", xp.sum(xp.abs(rand_mat) ** 2))
print("mean", xp.mean(rand_mat))
slicevis = SlicingVisualizer(rand_mat)
slicevis.run()
# %%
discard = mus.aperture_synthesis.discard_higher_kz(rand_mat, 30)
print("abs", xp.sum(xp.abs(discard) ** 2) * 100 / 40)
print("mean", xp.mean(discard))
slicevis = SlicingVisualizer(xp.abs(discard))
slicevis.run()

# %%
pad_discard = mus.aperture_synthesis.zeropad_higher_kz(discard, 30)
print("abs", xp.sum(xp.abs(pad_discard) ** 2))
print("mean", xp.mean(pad_discard))

slicevis = SlicingVisualizer(xp.abs(pad_discard))
slicevis.run()
# %%
