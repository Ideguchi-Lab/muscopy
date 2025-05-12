import os
from importlib.util import find_spec
from typing import NewType, Union

gpu_on = os.environ.get("MUSCOPY_GPU", "False") == "True"

_cp = gpu_on and bool(find_spec("cupy"))


Region = NewType("Region", tuple[tuple[int, int], tuple[int, int]])
Regions = NewType("Regions", list[Region])
OffsetRegions = Union[Regions, None]
MIPRegion = Union[Region, None]

EDGE_SIZE = 0
OFFSET_REGS = None
MIP_CENTER = None


def set_edge_size(size: int):
    global EDGE_SIZE
    EDGE_SIZE = size


def set_offset_regs(offset_regs: OffsetRegions):
    global OFFSET_REGS
    OFFSET_REGS = offset_regs


def set_mip_center(center: MIPRegion):
    global MIP_CENTER
    MIP_CENTER = center


def print_backend():
    print("Using cupy" if _cp else "Using numpy")


class ArrayPrecision:
    """A class to define the precision of arrays used in the library.

    Attributes
    ----------
    int_length : int
        The number of bits used for integer arrays.
    float_length : int
        The number of bits used for float arrays.
    """

    def __init__(self, int_length: int = 64, float_length: int = 64) -> None:
        """Construct an ArrayPrecision object with specified integer and float lengths.

        Parameters
        ----------
        int_length : int, optional
            The number of bits used for integer arrays, by default 64
        float_length : int, optional
            The number of bits used for float arrays, by default 64
        """
        self.int_length = int_length
        self.float_length = float_length

    def get_float_precision(self):
        return f"float{self.float_length}"

    def get_int_precision(self):
        return f"int{self.int_length}"

    def get_complex_precision(self):
        return f"complex{2 * self.float_length}"
