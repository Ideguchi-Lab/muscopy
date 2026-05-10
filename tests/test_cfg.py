"""Tests for configuration helpers."""

import jax
import pytest

from muscopy.cfg import ArrayPrecision


def test_array_precision_defaults_to_portable_32_bit() -> None:
    precision = ArrayPrecision()

    assert precision.int_precision() == "int32"
    assert precision.float_precision() == "float32"
    assert precision.complex_precision() == "complex64"


def test_array_precision_rejects_64_bit_when_jax_x64_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax.config, "read", lambda name: False if name == "jax_enable_x64" else None)

    with pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"):
        ArrayPrecision(int_length=32, float_length=64)

    with pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"):
        ArrayPrecision(int_length=64, float_length=32)


def test_array_precision_allows_64_bit_when_jax_x64_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax.config, "read", lambda name: True if name == "jax_enable_x64" else None)

    precision = ArrayPrecision(int_length=64, float_length=64)

    assert precision.int_precision() == "int64"
    assert precision.float_precision() == "float64"
    assert precision.complex_precision() == "complex128"


@pytest.mark.parametrize(
    ("int_length", "float_length", "message"),
    [
        (8, 32, "int_length must be one of 16, 32, or 64"),
        (32, 16, "float_length must be one of 32 or 64"),
    ],
)
def test_array_precision_rejects_unsupported_bit_lengths(
    int_length: int,
    float_length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ArrayPrecision(int_length=int_length, float_length=float_length)
