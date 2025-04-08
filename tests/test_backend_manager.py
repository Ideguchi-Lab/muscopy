import importlib.util
import types

import numpy as np
import pytest

import muscopy as mus
from muscopy.backend_manager import BackendManager


def dummy_cupy_module() -> types.ModuleType:
    """Create a dummy cupy module for testing purposes.

    Returns
    -------
    types.ModuleType
        A dummy cupy module with a minimal implementation.
    """
    dummy = types.ModuleType("cupy")
    dummy.array = lambda x: x
    return dummy


def test_initial_backend_property() -> None:
    """Test that the initial backend property is 'numpy'."""
    bm = BackendManager()
    assert bm.backend == "numpy"


def test_use_numpy_sets_backend() -> None:
    """Test that use_numpy() correctly sets __backend to 'numpy'."""
    bm = BackendManager()
    # Simulate a scenario where __backend is not 'numpy'
    bm.use_numpy()
    assert bm.backend == "numpy"


def test_get_backend_numpy() -> None:
    """Test that get_backend() returns the numpy module when __backend is 'numpy'."""
    bm = BackendManager()
    bm.use_numpy()
    module = bm.get_backend()

    assert module.__name__ == np.__name__


def test_use_cupy_not_available(monkeypatch) -> None:  # noqa: ANN001
    """
    Test that use_cupy() raises an ImportError when cupy is not available.

    This is achieved by monkeypatching importlib.util.find_spec to simulate that
    cupy cannot be found.
    """
    bm = BackendManager()

    # Override find_spec to simulate cupy is not available
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        mus.backend_manager, "find_spec", lambda name: False if name == "cupy" else original_find_spec(name)
    )
    with pytest.raises(ImportError) as excinfo:
        bm.use_cupy()
    assert "Cupy is not available" in str(excinfo.value)
