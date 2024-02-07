import os

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from tqdm import tqdm

from src.qpi import make_disk

EDGE_SIZE = 2  # for avoiding edge artifact in ifft


def find_max_args(array):
    max_value = xp.max(array)
    max_idx = xp.unravel_index(np.argmax(array), array.shape)
    max_x = max_idx[0]
    max_y = max_idx[1]

    return max_x, max_y, max_value


def preprocess_for_synthesis(array, ref_array=None, params=None, load_fft=False):
    global EDGE_SIZE
    REF_REGIONS = [
        [[2, 20], [2, 20]],
        [
            [2, 20],
            [params.aperturesize - 20 - EDGE_SIZE, params.aperturesize - 2 - EDGE_SIZE],
        ],
        [
            [params.aperturesize - 20 - EDGE_SIZE, params.aperturesize - 2 - EDGE_SIZE],
            [2, 20],
        ],
        [
            [params.aperturesize - 20 - EDGE_SIZE, params.aperturesize - 2 - EDGE_SIZE],
            [params.aperturesize - 20 - EDGE_SIZE, params.aperturesize - 2 - EDGE_SIZE],
        ],
    ]
    if load_fft:
        array_fft = array
    else:
        norm_array = array * params.pixelsize
        norm_array_fft = xp.fft.fftshift(xp.fft.fft2(norm_array, norm="ortho"))
        array_fft = norm_array_fft / params.k_unit

    disk = make_disk(params.offaxis_center, params.aperturesize // 2, array_fft.shape)
    array_fft = array_fft * disk
    max_x, max_y, _ = find_max_args(np.abs(array_fft))
    oblique_center = (
        max_x - params.offaxis_center[0],
        max_y - params.offaxis_center[1],
    )

    left_index = max_x - params.aperturesize
    right_index = max_x + params.aperturesize + 1
    top_index = max_y - params.aperturesize
    bottom_index = max_y + params.aperturesize + 1

    array_fft_pad = xp.pad(
        array_fft,
        (
            (params.aperturesize, params.aperturesize),
            (params.aperturesize, params.aperturesize),
        ),
        mode="constant",
        constant_values=0,
    )

    array_fft = array_fft_pad[
        left_index + params.aperturesize : right_index + params.aperturesize,
        top_index + params.aperturesize : bottom_index + params.aperturesize,
    ]

    norm_array_fft = array_fft * params.k_unit
    norm_array_cropped = xp.fft.ifft2(xp.fft.ifftshift(norm_array_fft), norm="ortho")[
        EDGE_SIZE:, EDGE_SIZE:
    ]
    array_cropped = norm_array_cropped / params.imgpx_unit

    if not ref_array is None:
        if load_fft:
            ref_array_fft = ref_array
        else:
            norm_ref_array = ref_array * params.pixelsize
            norm_ref_array_fft = xp.fft.fftshift(
                xp.fft.fft2(norm_ref_array, norm="ortho")
            )
            ref_array_fft = norm_ref_array_fft / params.k_unit
        ref_array_fft = ref_array_fft * disk
        ref_array_fft_pad = xp.pad(
            ref_array_fft,
            (
                (params.aperturesize, params.aperturesize),
                (params.aperturesize, params.aperturesize),
            ),
            mode="constant",
            constant_values=0,
        )
        ref_array_fft = ref_array_fft_pad[
            left_index + params.aperturesize : right_index + params.aperturesize,
            top_index + params.aperturesize : bottom_index + params.aperturesize,
        ]
        norm_ref_array_fft = ref_array_fft * params.k_unit
        norm_ref_array_cropped = xp.fft.ifft2(
            xp.fft.ifftshift(norm_ref_array_fft), norm="ortho"
        )[EDGE_SIZE:, EDGE_SIZE:]
        ref_array_cropped = norm_ref_array_cropped / params.imgpx_unit

    if not ref_array is None:
        array_divided = array_cropped / ref_array_cropped
    else:
        array_divided = array_cropped

    phase_offset_list = []
    amplitude_offset_list = []
    for region in REF_REGIONS:
        phase_offset_list.append(
            xp.mean(
                xp.angle(
                    array_divided[
                        region[0][0] : region[0][1], region[1][0] : region[1][1]
                    ]
                )
            )
        )
        amplitude_offset_list.append(
            xp.mean(
                xp.abs(
                    array_divided[
                        region[0][0] : region[0][1], region[1][0] : region[1][1]
                    ]
                )
            )
        )
    phase_offset = xp.mean(xp.array(phase_offset_list))
    amplitude_offset = xp.mean(xp.array(amplitude_offset_list))

    array_divided = array_divided * xp.exp(-1j * phase_offset)
    array_divided = array_divided / amplitude_offset

    return array_divided, oblique_center


class Synthesizer:
    def __init__(self):
        pass

    def set_parameters(self, params):
        self.params = params
        self.params.calc_params()

    def set_data(self, target, reference=None):
        self.target_data = target
        self.reference_data = reference

        self.target_data.sort()
        if self.reference_data is not None:
            self.reference_data.sort()

    def load_oblique_centers(self, centers_path):
        """will be deprecated"""
        self.oblique_centers = []
        with open(centers_path, "r") as f:
            for line in f:
                self.oblique_centers.append(tuple(map(int, line.split(","))))

    def synthesize(self, save_multiangle=False, load_fft=False):
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
            ),
            dtype=xp.complex128,
        )
        synthesized_center = (
            self.params.aperturesize - EDGE_SIZE // 2,
            self.params.aperturesize - EDGE_SIZE // 2,
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        self.multiangle_qpi = dict()

        print("synthesizing...")
        for i in tqdm(range(len(self.target_data))):
            array = xp.load(self.target_data[i])

            if self.reference_data is not None:
                ref_array = xp.load(self.reference_data[i])

                array_cropped, oblique_center = preprocess_for_synthesis(
                    array, ref_array, params=self.params, load_fft=load_fft
                )

            else:
                array_cropped, oblique_center = preprocess_for_synthesis(
                    array, params=self.params, load_fft=load_fft
                )
            disk_synthesized = make_disk(
                (
                    synthesized_center[0] - oblique_center[0],
                    synthesized_center[1] - oblique_center[1],
                ),
                self.params.aperturesize // 2,
                array_cropped.shape,
            )

            if save_multiangle:
                self.multiangle_qpi[i] = xp.angle(array_cropped)
                # self.multiangle_qpi[i] = xp.imag(array_cropped)

            norm_array_cropped = array_cropped * self.params.imgpx_unit
            norm_fft_cropped = xp.fft.fftshift(
                xp.fft.fft2(norm_array_cropped, norm="ortho")
            )
            fft_cropped = norm_fft_cropped / self.params.k_unit

            fft_cropped = fft_cropped * disk_synthesized

            synthesized_fft += fft_cropped
            synthesized_weight += disk_synthesized != 0

        synthesized_fft /= synthesized_weight
        norm_synthesized_fft = synthesized_fft * self.params.k_unit
        norm_synthesized_array = xp.fft.ifft2(
            xp.fft.ifftshift(norm_synthesized_fft), norm="ortho"
        )
        synthesized_array = norm_synthesized_array / self.params.imgpx_unit
        synthesized_qpi = xp.angle(synthesized_array)
        # synthesized_qpi = xp.abs(xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft)))

        if _cp:
            synthesized_fft = xp.asnumpy(synthesized_fft)
            synthesized_qpi = xp.asnumpy(synthesized_qpi)

        # synthesized_qpi = unwrap_phase(synthesized_qpi)

        if save_multiangle:
            if os.path.exists("multiangle_qpi"):
                import shutil

                shutil.rmtree("multiangle_qpi")
            os.mkdir("multiangle_qpi")
            for i in range(len(self.target_data)):
                # save as png
                if _cp:
                    plt.imsave(
                        f"multiangle_qpi/{i:03}.png",
                        xp.asnumpy(self.multiangle_qpi[i]),
                        cmap="gray",
                    )
                else:
                    plt.imsave(
                        f"multiangle_qpi/{i:03}.png",
                        self.multiangle_qpi[i],
                        cmap="gray",
                    )

        return synthesized_qpi, synthesized_fft
