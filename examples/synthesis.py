# %%
# import libraries
import os
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
from src.aperture_synthesis import Synthesizer

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

backend = "cupy" if _cp else "numpy"
print("Using {} backend".format(backend))

# %%

TARGET_PATH = r"data/aperture_sample/"
REF_PATH = r"data/aperture_ref/"

synthesizer = Synthesizer()
synthesizer.set_parameters(
    wavelength=532 * 10 ** (-9),
    NA=1.2,
    img_shape=(2048, 2048),
    img_center=(1024, 1024),
    pixelsize=3.45 * 1e-6 * 3 / 200,
    offaxis_center=(562, 1656),
)

synthesizer.set_path(TARGET_PATH, REF_PATH)
c, r, res = synthesizer.search_centers()
print("true center:", c)
print("radius:", r)
print("residual:", res)
synthesizer.load_oblique_centers(os.path.join(TARGET_PATH, "centers.txt"))
qpi, fft = synthesizer.synthesize()

# %%

print(qpi.shape)
fig, ax = plt.subplots()
im = ax.imshow(qpi)
pp = fig.colorbar(im, ax=ax)
im.set_clim(0, 1)

pp.set_label("phase")
# plt.show()

# plt.savefig("synthesized_beads.png", dpi=500)
# np.save("synthesized.npy", synthesized_qpi)


# %%

fig, ax = plt.subplots()
im = ax.imshow(np.log(np.abs(fft)))
fig.colorbar(im, ax=ax)
plt.show()


# %%
