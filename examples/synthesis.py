# %%
# import libraries
import os
import sys

sys.path.append("../..")

import numpy as np
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from microscopy_converters.src.aperture_synthesis import Synthesizer
from microscopy_converters.src.qpi import QPIParameters, make_disk
from PIL import Image
from skimage.restoration import unwrap_phase

try:
    import cupy as xp
    _cp = True
except ImportError:
    import numpy as xp
    _cp = False

backend = "cupy" if _cp else "numpy"
print("Using {} backend".format(backend))

TARGET_PATH =r"C:\Users\fukus\fukushima-dir\git-repo\dev-realimage\experiments\aperture\aperture20230828_114455/"
# TARGET_PATH = r"C:\Users\fukus\fukushima-dir\git-repo\dev-realimage\experiments\aperture\20230729\try2/"
REF_PATH = r"C:\Users\fukus\fukushima-dir\git-repo\dev-realimage\experiments\aperture\aperture20230828_114759/"
# REF_PATH = r"C:\Users\fukus\fukushima-dir\git-repo\dev-realimage\experiments\aperture\20230729\try2\ref/"

# read centers from file
oblique_centers = []
with open(os.path.join(TARGET_PATH, "centers.txt"), "r") as f:
    for line in f:
        center = tuple(map(int, line.split(",")))
        oblique_centers.append(center)

print(oblique_centers)

# synthesized_fft = xp.zeros((1001, 1001), dtype=np.complex128)
# synthesized_center = (500, 500)
synthesized_fft = xp.zeros((1001, 1001), dtype=np.complex128)
# synthesized_fft = xp.zeros((201, 201), dtype=np.complex128)
synthesized_center = (500, 500)
# synthesized_center = (100, 100)
# synthesized_aperture_length = 100
synthesized_aperture_length = 500
synthesized_weight = xp.ones(synthesized_fft.shape)

def find_max_args(array_fft):
    max_args = xp.unravel_index(xp.argmax(xp.abs(array_fft)), array_fft.shape)
    return (max_args[0], max_args[1])

def calc_phase_offset(array_fft, synthesized_fft, synthesized_weight, synthesized_center):
    # map array into expanded space
    # synthesized = xp.array(synthesized_fft)
    synthesized = xp.copy(synthesized_fft) / synthesized_weight
    overlap_region = (array_fft != 0) * (synthesized != 0)
    phase_offset = xp.mean(xp.angle(synthesized[overlap_region] / array_fft[overlap_region]))
    if _cp:
        phase_offset = xp.asnumpy(phase_offset)
    return phase_offset

def mean_with_weight(array1, array2):
    """Take mean of two arrays with weight.

    Args:
        array1 (_type_): _description_
        array2 (_type_): _description_
    """
    weight1 = xp.abs(array1)
    weight2 = xp.abs(array2)
    weight = weight1 + weight2
    weight[weight == 0] = 1
    return (array1 + array2) / (weight1 + weight2)

# %%

# set qpi parameter
params = QPIParameters(
    wavelength=532 * 10 ** (-9),
    NA=1.2,
    img_shape=(2048, 2048),
    img_center=(1024, 1024),
    pixelsize=3.45 * 1e-6 * 3 / 200,
    # center=(580, 623),
    # center = (522, 651),
    center=(561, 1657)
)
params.calc_params()
print(params.aperturesize)

c_center = (params.aperturesize//2, params.aperturesize//2)


# %%
pathlist = os.listdir(TARGET_PATH)
pic_path_list = []

refpathlist = os.listdir(REF_PATH)
ref_pic_path_list = []

for i in range(len(pathlist)):
    filename = pathlist[i]
    if filename.endswith(".png"):
        pic_path_list.append(filename)

for i in range(len(refpathlist)):
    filename = refpathlist[i]
    if filename.endswith(".png"):
        ref_pic_path_list.append(filename)

pic_path_list = sorted(pic_path_list)
ref_pic_path_list = sorted(ref_pic_path_list)
print(pic_path_list)
print(ref_pic_path_list)

# %%

pic_count = 0
ref_pic_count = 0 
# for i in tqdm(range(len(pathlist))):
for i in range(len(pic_path_list)):
    filename = pic_path_list[i]
    ref_filename = ref_pic_path_list[i]
    pic_count += 1
    ref_pic_count += 1
    print(filename)

    image = Image.open(TARGET_PATH + filename)
    ref_image = Image.open(REF_PATH + ref_filename)

    array = xp.array(image).reshape(2048, 2048)[:2048, :2048]
    ref_array = xp.array(ref_image).reshape(2048, 2048)[:2048, :2048]

    fft = xp.fft.fftshift(xp.fft.fft2(array))
    ref_fft = xp.fft.fftshift(xp.fft.fft2(ref_array))

    oblique_center = oblique_centers[pic_count-1]
    print(oblique_center)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(xp.asnumpy(xp.log(xp.abs(fft))))
    ax[1].imshow(xp.asnumpy(xp.log(xp.abs(ref_fft))))
    ax[0].scatter(params.center[0] + oblique_center[1], params.center[1] + oblique_center[0], s=10, c="r")
    ax[1].scatter(params.center[0] + oblique_center[1], params.center[1] + oblique_center[0], s=10, c="r")
    # fig, ax = plt.subplots()
    # ax.imshow(xp.asnumpy(xp.log(xp.abs(fft))))
    # ax.scatter(params.center[0] + oblique_center[1], params.center[1] + oblique_center[0], s=10, c="r")
    plt.show()

    disk = make_disk(params.center, params.aperturesize/2,  fft.shape)

    fft = fft * disk
    ref_fft = ref_fft * disk

    left_index = params.center[1] + oblique_center[0] - synthesized_aperture_length
    right_index = params.center[1] + oblique_center[0] + synthesized_aperture_length + 1
    top_index = params.center[0] + oblique_center[1] - synthesized_aperture_length
    bottom_index = params.center[0] + oblique_center[1] + synthesized_aperture_length + 1
    if left_index < 0:
        left_index = 0
    if right_index > params.img_shape[0]:
        right_index = params.img_shape[0]
    if top_index < 0:
        top_index = 0
    if bottom_index > params.img_shape[1]:
        bottom_index = params.img_shape[1]

    fft = fft[
            left_index: right_index,
            top_index: bottom_index,
        ]
    ref_fft = ref_fft[
            left_index: right_index,
            top_index: bottom_index,
        ]
    
    if params.center[1] + oblique_center[0] - synthesized_aperture_length < 0:
            fft = xp.pad(fft, ((-params.center[1] - oblique_center[0] + synthesized_aperture_length, 0), (0, 0)), "constant")
            ref_fft = xp.pad(ref_fft, ((-params.center[1] - oblique_center[0] + synthesized_aperture_length, 0), (0, 0)), "constant")
            # print(1)
            # print(fft.shape)
            # print(ref_fft.shape)
    if params.center[1] + oblique_center[0] + synthesized_aperture_length + 1 > params.img_shape[0]:
            fft = xp.pad(fft, ((0, params.center[1] + oblique_center[0]
            + synthesized_aperture_length
            + 1 - params.img_shape[0]), (0, 0)), "constant")
            ref_fft = xp.pad(ref_fft, ((0, params.center[1] + oblique_center[0]
            + synthesized_aperture_length
            + 1 - params.img_shape[0]), (0, 0)), "constant")
            # print(2)
            # print(fft.shape)
            # print(ref_fft.shape)
    if params.center[0] + oblique_center[1] - synthesized_aperture_length < 0:
            fft = xp.pad(fft, ((0, 0), (-params.center[0] - oblique_center[1] + synthesized_aperture_length, 0)), "constant")
            ref_fft = xp.pad(ref_fft, ((0, 0), (-params.center[0] - oblique_center[1] + synthesized_aperture_length, 0)), "constant")
            # print(3)
            # print(fft.shape)
            # print(ref_fft.shape)
    if params.center[0] + oblique_center[1] + synthesized_aperture_length + 1 > params.img_shape[1]:
            fft = xp.pad(fft, ((0, 0), (0, params.center[0] + oblique_center[1]
            + synthesized_aperture_length
            + 1 - params.img_shape[1])), "constant")
            ref_fft = xp.pad(ref_fft, ((0, 0), (0, params.center[0] + oblique_center[1]
            + synthesized_aperture_length
            + 1 - params.img_shape[1])), "constant")
            # print(4)
            # print(fft.shape)
            # print(ref_fft.shape)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(xp.asnumpy(xp.log(xp.abs(fft))))
    # ax[1].imshow(xp.asnumpy(xp.log(xp.abs(ref_fft))))
    # ax[0].scatter(synthesized_center[1], synthesized_center[0], c="r")
    # ax[1].scatter(synthesized_center[1], synthesized_center[0], c="r")
    # fig, ax = plt.subplots()
    # ax.imshow(xp.asnumpy(xp.log(xp.abs(fft))))
    # ax.scatter(synthesized_center[1], synthesized_center[0], c="r")
    # plt.show()

    qpi_array = xp.fft.ifft2(xp.fft.ifftshift(fft))
    ref_qpi_array = xp.fft.ifft2(xp.fft.ifftshift(ref_fft))
    qpi_divided = qpi_array / ref_qpi_array

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(xp.asnumpy((xp.angle(qpi_array))))
    ax[1].imshow(xp.asnumpy((xp.angle(ref_qpi_array))))
    plt.show() 

    # normalize
    background_region = ((0, 50), (0, 50))
    phase_backgound = xp.mean(xp.angle(qpi_divided[background_region[0][0]:background_region[0][1], background_region[1][0]:background_region[1][1]]))

    amplitude_background = xp.mean(xp.abs(qpi_divided[background_region[0][0]:background_region[0][1], background_region[1][0]:background_region[1][1]]))

    qpi_divided = qpi_divided / amplitude_background

    qpi_divided = qpi_divided * xp.exp(-1j * phase_backgound)

    fft_divided = xp.fft.fftshift(xp.fft.fft2(qpi_divided))

    # print("check cropped qpi")
    # qpi_croped = xp.fft.ifft2(xp.fft.ifftshift(fft_divided))
    # fig, ax = plt.subplots()
    # ax.imshow(xp.asnumpy((xp.angle(qpi_croped))))
    # plt.show()

    disk_synthesized = make_disk((synthesized_center[0] - oblique_center[1], synthesized_center[1] - oblique_center[0]), params.aperturesize//2, synthesized_fft.shape)
    print(fft_divided.shape)

    fft_divided = fft_divided * disk_synthesized

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(xp.asnumpy(xp.log(xp.abs(fft_divided))))
    ax.scatter(synthesized_center[1], synthesized_center[0], c="r")
    ax.scatter(synthesized_center[1] - oblique_center[1], synthesized_center[0] +
               - oblique_center[0], c="b")
    # fig.colorbar(im, ax=ax)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(xp.asnumpy(xp.angle(qpi_divided))) 
    # fig.colorbar(im, ax=ax)
    plt.show()

    if pic_count > 1:
        phase_offset = calc_phase_offset(fft_divided, synthesized_fft, synthesized_weight, synthesized_center)
        fft_divided = fft_divided * xp.exp(1j * phase_offset)

    synthesized_fft += fft_divided
    synthesized_weight += (disk_synthesized != 0)
    # synthesized_fft = mean_with_weight(synthesized_fft, fft_divided)
        
    print("reconstructed")
    fig, ax = plt.subplots()
    im = ax.imshow(xp.asnumpy(xp.angle(xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft/ synthesized_weight)))))
    # im = ax.imshow(xp.asnumpy(xp.angle(xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft)))))
    # fig.colorbar(im, ax=ax)
    plt.show()
        

synthesized_fft /= synthesized_weight

print("number of pictures: {}".format(pic_count))
fig, ax = plt.subplots()
im = ax.imshow(xp.asnumpy(synthesized_weight))
pp = fig.colorbar(im, ax=ax)
pp.set_clim(0, 1)
pp.set_label("phase")
plt.show()

# %%

fig, ax = plt.subplots()
im = ax.imshow(xp.asnumpy(xp.log(xp.abs(synthesized_fft))))
fig.colorbar(im, ax=ax)
plt.show()

# %%
# inverse fourier transform
synthesized_array = xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft))
synthesized_qpi = xp.angle(synthesized_array)

if _cp:
    synthesized_fft = xp.asnumpy(synthesized_fft)
    # synthesized_array = xp.asnumpy(synthesized_array)
    synthesized_qpi = xp.asnumpy(synthesized_qpi)

# synthesized_qpi = unwrap_phase(synthesized_qpi)

fig, ax = plt.subplots()
# im = ax.imshow(unwrap_phase(synthesized_qpi))
im = ax.imshow(synthesized_qpi)
pp = fig.colorbar(im, ax=ax)
# pp.set_clim(0, 1)
# limit color map range into [0, 1]
im.set_clim(0, 1)

pp.set_label("phase")
# plt.show()

# plt.savefig("synthesized.png", dpi=500)
# np.save("synthesized.npy", synthesized_qpi)
print(synthesized_qpi.shape)hjg


# %%

# resolution 2048x2048
fig, ax = plt.subplots()
im = ax.imshow(xp.asnumpy(xp.log(xp.abs(synthesized_fft))))
fig.colorbar(im, ax=ax)
plt.show()


# %%
