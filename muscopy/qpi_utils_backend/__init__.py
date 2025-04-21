"""QPI utilities backend."""

import importlib.util

from . import qpi_util_cpu  # noqa: F401

if importlib.util.find_spec(".qpi_util_gpu", package=__package__):
    import qpi_util_gpu  # noqa: F401
