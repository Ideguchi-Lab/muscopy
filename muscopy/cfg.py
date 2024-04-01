from typing import NewType, Union

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
