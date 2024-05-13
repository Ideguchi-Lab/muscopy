import os
from typing import NewType, Union

gpu_on = os.environ.get("MUSCOPY_GPU", "False") == "True"

if gpu_on:
    try:
        import cupy as xp

        _cp = True
    except ImportError:
        import numpy as xp

        _cp = False
else:
    import numpy as xp

    _cp = False


Regions = NewType("Regions", list[tuple[tuple[int, int], tuple[int, int]]])
OffsetRegions = Union[Regions, None]

EDGE_SIZE = 0
OFFSET_REGS = None


def set_edge_size(size: int):
    global EDGE_SIZE
    EDGE_SIZE = size


def set_offset_regs(offset_regs: OffsetRegions):
    global OFFSET_REGS
    OFFSET_REGS = offset_regs


class ArrayPrecision:
    def __init__(self, int_length: int, float_length: int):
        self.int_length = int_length
        self.float_length = float_length

    def get_float_precision(self):
        return f"float{self.float_length}"

    def get_int_precision(self):
        return f"int{self.int_length}"

    def get_complex_precision(self):
        return f"complex{2 * self.float_length}"
