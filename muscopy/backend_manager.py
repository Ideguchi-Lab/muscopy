"""Backend manager for Muscopy.

This module provides:

- `BackendManager`: A class to manage the backend for Muscopy.
- `ArrayProtocol`: A protocol for array-like objects.
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Protocol, TypeVar, overload, runtime_checkable

if TYPE_CHECKING:
    import types

    from numpy.typing import DTypeLike


class BackendManager:
    """Backend manager for Muscopy."""

    def __init__(self) -> None:
        self.__backend = "numpy"

    @property
    def backend(self) -> str:
        """Get the current backend."""
        return self.__backend

    def use_numpy(self) -> None:
        """Set the backend to numpy."""
        self.__backend = "numpy"

    def use_cupy(self) -> None:
        """Set the backend to cupy.

        Raises
        ------
        ImportError
            If cupy is not available.
        """
        if not bool(find_spec("cupy")):
            msg = "Cupy is not available. Please install cupy to use the GPU backend."
            raise ImportError(msg)
        self.__backend = "cupy"

    def get_backend(self) -> types.ModuleType:
        """Get the current backend module.

        Returns
        -------
        types.ModuleType
            The current backend module (numpy or cupy).

        Raises
        ------
        ValueError
            If the backend is not numpy or cupy.
        """
        if self.__backend == "numpy":
            import numpy as np  # noqa: PLC0415

            return np
        if self.__backend == "cupy":
            import cupy as cp  # noqa: PLC0415

            return cp
        msg = f"Unexpected backend: {self.__backend}"
        raise ValueError(msg)


# Generic type variable for array elements (covariant)
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ArrayProtocol(Protocol[T_co]):
    """
    A protocol for array-like objects that supports both NumPy and CuPy arrays.

    Array-like objects must have the following attributes:
      - shape: a tuple of integers representing the dimensions.
      - dtype: the data type of the elements (as a numpy.dtype).
      - ndim: the number of dimensions.
      - size: the total number of elements.

    In addition, they should implement:
      - A read-only property `T` returning the transposed array.
      - The __getitem__ method to allow indexing, with overloads for different index types.
      - The __array__ method to convert the object to a NumPy ndarray.
      - The astype method to cast the array to a specified dtype.
      - The reshape method to return a reshaped version of the array.
      - The copy method to return a copy of the array.
      - The sum method to compute the sum of array elements, with overloads based on the axis parameter.

    This protocol is designed to be used in environments where both NumPy and CuPy arrays are relevant.
    In environments where CuPy is not available, only NumPy arrays may be used.
    """

    # Attributes
    shape: tuple[int, ...]
    dtype: DTypeLike
    ndim: int
    size: int

    @property
    def T(self) -> ArrayProtocol[T_co]:  # noqa: N802
        """Read-only property returning the transposed array."""
        ...

    # __getitem__ overloads based on the type of index (int, slice, or tuple of ints)
    @overload
    def __getitem__(self, index: int) -> T_co:
        ...

    @overload
    def __getitem__(self, index: slice) -> ArrayProtocol[T_co]:
        ...

    @overload
    def __getitem__(self, index: tuple[int, ...]) -> ArrayProtocol[T_co]:
        ...

    def __getitem__(self, index: int | slice | tuple[int, ...]) -> T_co | ArrayProtocol[T_co]:
        ...

    def __array__(self) -> ArrayProtocol[T_co]:  # noqa: PLW3201
        """Convert the object to a NumPy ndarray."""
        ...

    def astype(self, dtype: DTypeLike) -> ArrayProtocol[T_co]:
        """Return a copy of the array cast to the specified dtype."""
        ...

    def reshape(self, shape: tuple[int, ...]) -> ArrayProtocol[T_co]:
        """Return a reshaped view of the array."""
        ...

    def copy(self) -> ArrayProtocol[T_co]:
        """Return a copy of the array."""
        ...

    @overload
    def sum(self, axis: None = None) -> T_co:
        ...

    @overload
    def sum(self, axis: int) -> ArrayProtocol[T_co]:
        ...

    @overload
    def sum(self, axis: tuple[int, ...]) -> ArrayProtocol[T_co]:
        ...

    def sum(self, axis: int | tuple[int, ...] | None = None) -> T_co | ArrayProtocol[T_co]:
        """
        Return the sum of the array elements over a given axis.

        If `axis` is None, returns a scalar of type T_co.
        Otherwise, returns an array-like object with the summed values.
        """
        ...
