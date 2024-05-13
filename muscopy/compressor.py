import pickle

import muscopy as mus
import muscopy.cfg as mcfg

from .aperture_synthesis import DataHolder, Params, get_oblique_field
from .dir_parser import numpy_parser

if mcfg._cp:
    import cupy as xp
else:
    import numpy as xp


class HologramCompressor:
    def __init__(self, params: Params):
        self.params = params

    def load_holograms(self, hologram_dir_path: str):
        self.hologram_paths = numpy_parser(hologram_dir_path)

    def compress(self) -> DataHolder:
        shape = (
            2 * self.params.aperturesize + 1 - mus.cfg.EDGE_SIZE,
            2 * self.params.aperturesize + 1 - mus.cfg.EDGE_SIZE,
        )
        field = xp.zeros(shape, dtype=xp.complex128)
        for hologram_path in self.hologram_paths:
            hologram = xp.load(hologram_path)
            oblique_field, oblique_shift = get_oblique_field(hologram, self.params)
            field += oblique_field

        data = DataHolder()
        data.sample_field = field
        data.oblique_shift = oblique_shift
        return data

    @classmethod
    def save_as_pickle(cls, data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
