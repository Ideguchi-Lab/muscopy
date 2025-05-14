"""QPI utilities backend."""

import importlib.util

from . import qpi_util_cpu  # noqa: F401

if importlib.util.find_spec("cupy") is not None:
    from . import qpi_util_gpu  # noqa: F401
