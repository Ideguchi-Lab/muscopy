"""Configuration module for microscopy converters."""

from typing import TypeAlias

import jax

Region: TypeAlias = tuple[tuple[int, int], tuple[int, int]]
Regions: TypeAlias = list[Region]
OffsetRegions: TypeAlias = Regions | None
MIPRegion: TypeAlias = Region | None

X64_BIT_LENGTH = 64

__all__ = [
    "ArrayPrecision",
    "MIPRegion",
    "OffsetRegions",
    "Region",
    "Regions",
]


def _jax_x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))  # type: ignore[no-untyped-call]


class ArrayPrecision:
    """A class to define the precision of arrays used in the library.

    The default precision is portable across standard JAX environments:
    ``int32`` for integers, ``float32`` for floats, and ``complex64`` for
    complex arrays. Requesting 64-bit arrays requires ``jax_enable_x64=True``.

    Attributes
    ----------
    int_length : `int`
        The number of bits used for integer arrays.
    float_length : `int`
        The number of bits used for float arrays.
    """

    def __init__(self, int_length: int = 32, float_length: int = 32) -> None:
        self.int_length = int_length
        self.float_length = float_length
        self.validate()

    def validate(self) -> None:
        """Validate the precision against supported JAX dtypes and x64 settings.

        Raises
        ------
        ValueError
            If the requested bit lengths are unsupported, or if 64-bit precision
            is requested while JAX x64 support is disabled.
        """
        if self.int_length not in {16, 32, 64}:
            msg = "int_length must be one of 16, 32, or 64."
            raise ValueError(msg)
        if self.float_length not in {32, 64}:
            msg = "float_length must be one of 32 or 64."
            raise ValueError(msg)
        if X64_BIT_LENGTH in {self.int_length, self.float_length} and not _jax_x64_enabled():
            msg = "64-bit precision requires JAX x64 support. Enable jax_enable_x64 before creating ArrayPrecision."
            raise ValueError(msg)

    def float_precision(self) -> str:
        """Return the precision of float arrays as a string.

        Returns
        -------
        `str`
            The precision of float arrays, e.g., "float32" or "float64".
        """
        self.validate()
        return f"float{self.float_length}"

    def int_precision(self) -> str:
        """Return the precision of integer arrays as a string.

        Returns
        -------
        `str`
            The precision of integer arrays, e.g., "int32" or "int64".
        """
        self.validate()
        return f"int{self.int_length}"

    def complex_precision(self) -> str:
        """Return the precision of complex arrays as a string.

        Returns
        -------
        `str`
            The precision of complex arrays, e.g., "complex64" or "complex128".
        """
        self.validate()
        return f"complex{2 * self.float_length}"
