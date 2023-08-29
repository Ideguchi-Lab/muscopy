# %%
from source.qpi import QPIParameters, decode_adimec, make_disk, qpi
from source.unwrap_phase import phase_unwrap
from skimage.restoration import unwrap_phase
from source.fft2d import fft_converter
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import time


try:
    import cupy as xp
    _cp = True
except:
    import numpy as xp
    _cp = False

backend = "numpy" if not _cp else "cupy"
print(f"Using {backend} backend")


# %%
# Load data
pil_array = Image.open(
    "source\Basler_acA2440-75um__22770929__20230303_161937637_0099.tiff")
pil_ref = Image.open(
    "source\Basler_acA2440-75um__22770929__20230303_162019973_0000 (1).tiff")

array = xp.array(pil_array)
ref = xp.array(pil_ref)

# %%

image_center = (511, 511)
array = array[:1023, :1023]
ref = ref[:1023, :1023]

params = QPIParameters(wavelength=532*10**(-9), NA=0.6,
                       img_shape=(1023, 1023), img_center=(512, 512), pixelsize=3.45*10**(-6)/40, center=(815, 209)
                       )
params.calc_params()
result = qpi(array, ref, params)
# result = fft_converter(array)
print(result.shape)

# %%
if _cp:
    result_np = xp.asnumpy(result)

plt.imshow(result_np, cmap="gray")
plt.colorbar()

# plt.scatter(815, 209)

# %%

unwrapped = phase_unwrap(result_np)

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
