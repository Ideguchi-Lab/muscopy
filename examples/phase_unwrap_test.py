# %%
import time

from matplotlib import pyplot as plt
from PIL import Image
from skimage.restoration import unwrap_phase

import muscopy as mus
from muscopy.qpi import QPIParameters, qpi

try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False

backend = "numpy" if not _cp else "cupy"
print(f"Using {backend} backend")

# %%
# config
mus.cfg.set_edge_size(0)
mus.cfg.set_offset_regs([((5, 15), (5, 15))])

# %%
# Load data
pil_array = Image.open(r"source\Basler_acA2440-75um__22770929__20230303_161937637_0099.tiff")
pil_ref = Image.open("source\Basler_acA2440-75um__22770929__20230303_162019973_0000 (1).tiff")

array = xp.array(pil_array)
ref = xp.array(pil_ref)

# %%

image_center = (511, 511)
array = array[:1023, :1023]
ref = ref[:1023, :1023]

params = mus.qpi.QPIParameters(
    wavelength=532 * 10 ** (-9),
    NA=0.6,
    img_shape=(1023, 1023),
    pixelsize=3.45 * 10 ** (-6) / 40,
    offaxis_center=(815, 209),
    n_sol=1.33,
)

result = mus.qpi.qpi(array, ref, params, mus.OFFSET_REGS)

# %%
if _cp:
    result_np = xp.asnumpy(result)

plt.imshow(result_np, cmap="gray")
plt.colorbar()

# plt.scatter(815, 209)

# %%

start = time.perf_counter()
unwrapped = mus.unwrap_phase.phase_unwrap(result_np)
print(time.perf_counter() - start)

# %%
plt.imshow(unwrapped)
plt.colorbar()

# %%

start = time.perf_counter()
unwrapped_ref = unwrap_phase(result_np)
print(time.perf_counter() - start)
plt.imshow(unwrapped_ref)
plt.colorbar()
# %%
