EDGE_SIZE = 2
OFFSET_REGS = [((0, 0), (0, 0))]


def set_edge_size(size: int):
    global EDGE_SIZE
    EDGE_SIZE = size


def set_offset_regs(offset_regs: list[tuple[tuple[int, int], tuple[int, int]]]):
    global OFFSET_REGS
    OFFSET_REGS = offset_regs
