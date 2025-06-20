"""Configuration module for microscopy converters."""

from typing import TypeAlias

Region: TypeAlias = tuple[tuple[int, int], tuple[int, int]]
Regions: TypeAlias = list[Region]
OffsetRegions: TypeAlias = Regions | None
MIPRegion: TypeAlias = Region | None


class ArrayPrecision:
    """A class to define the precision of arrays used in the library.

    Attributes
    ----------
    int_length : `int`
        The number of bits used for integer arrays.
    float_length : `int`
        The number of bits used for float arrays.
    """

    def __init__(self, int_length: int = 64, float_length: int = 64) -> None:
        self.int_length = int_length
        self.float_length = float_length

    def float_precision(self) -> str:
        """Return the precision of float arrays as a string.

        Returns
        -------
        `str`
            The precision of float arrays, e.g., "float32" or "float64".
        """
        return f"float{self.float_length}"

    def int_precision(self) -> str:
        """Return the precision of integer arrays as a string.

        Returns
        -------
        `str`
            The precision of integer arrays, e.g., "int32" or "int64".
        """
        return f"int{self.int_length}"

    def complex_precision(self) -> str:
        """Return the precision of complex arrays as a string.

        Returns
        -------
        `str`
            The precision of complex arrays, e.g., "complex64" or "complex128".
        """
        return f"complex{2 * self.float_length}"
