Quantitative Phase Imaging
===========================

QPI Calculator
++++++++++++++++++++++++++++++++

.. automodule:: muscopy.qpi
    :members:

QPI Utils
++++++++++++++++++++++++++++++++++++++++++

.. automodule:: muscopy.qpi_utils
    :members:

CuPy QPI
++++++++++++++++++++++++++++++++++++++++++

The optional CuPy implementation is available from the ``muscopy.cuda``
subpackage after installing the CUDA 12 extra:

.. code-block:: sh

    uv sync --extra cupy

It provides CuPy-based ``offaxis_dh``, ``qpi``, ``mip_qpi``,
``correct_phase_offset``, and ``unwrap_phase`` helpers without changing the
default JAX API.

.. automodule:: muscopy.cuda.dh
    :members:

.. automodule:: muscopy.cuda.qpi
    :members:

.. automodule:: muscopy.cuda.qpi_utils
    :members:
