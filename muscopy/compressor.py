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
        hologram_paths = numpy_parser(hologram_dir_path)
        self.holograms = [xp.load(hologram_path) for hologram_path in hologram_paths]

    def load_holograms_from_list(self, hologram_list: list):
        self.holograms = [xp.array(hologram) for hologram in hologram_list]

    def compress(self) -> DataHolder:
        shape = (
            2 * self.params.aperturesize + 1 - mus.cfg.EDGE_SIZE,
            2 * self.params.aperturesize + 1 - mus.cfg.EDGE_SIZE,
        )
        field = xp.zeros(shape, dtype=xp.complex128)
        for hologram in self.holograms:
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
