# %%
import os

os.environ["MUSCOPY_GPU"] = "False"

from ilabvis import CursorVisualizer
from PIL import Image

import muscopy as mus

if mus.cfg._cp:
    import cupy as xp
else:
    import numpy as xp


# %%
hologram_path = r"EDIT PATH"

params = mus.QPIParameters(
    wavelength=532e-9,
    NA=1.2,
    img_shape=(1024, 1024),
    pixelsize=3.45e-6 * 3 / 250,
    offaxis_center=(309, 244),
    n_sol=1.33,
)
params.print_all_parameters()

# %%
hologram = xp.array(Image.open(hologram_path))
visibility = mus.qpi.calc_visibility(hologram, params)
phase_noise = mus.qpi.calc_phase_noise(hologram, params)


# %%
CursorVisualizer(hologram).run()
CursorVisualizer(visibility).run()
CursorVisualizer(phase_noise).run()

# %%
