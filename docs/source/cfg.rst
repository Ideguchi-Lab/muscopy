Configuration
=========================

Precision Contract
------------------

``ArrayPrecision`` defaults to ``int32``, ``float32``, and ``complex64``.
These defaults match standard JAX environments where ``jax_enable_x64`` is
disabled.

If 64-bit integer, floating-point, or complex precision is required, enable
JAX x64 support before constructing ``ArrayPrecision``. Muscopy does not
change JAX's global x64 setting automatically. When 64-bit precision is
requested while x64 support is disabled, ``ArrayPrecision`` raises a
``ValueError`` instead of relying on JAX's implicit truncation to 32-bit dtypes.

Examples
--------

.. code-block:: python

    from muscopy.cfg import ArrayPrecision
    from muscopy.odt import EwaldEmbeddingMode, ODTConfig

    odt_config = ODTConfig(
        precision=ArrayPrecision(),
        ewald_embedding_mode=EwaldEmbeddingMode.TRUNCATE,
    )

.. code-block:: python

    import jax

    from muscopy.cfg import ArrayPrecision

    jax.config.update("jax_enable_x64", True)
    precision = ArrayPrecision(int_length=64, float_length=64)

ODT Reconstruction Options
--------------------------

``ODTConfig.gradient_correction`` defaults to ``True`` to preserve the existing
ODT reconstruction behavior. Set it to ``False`` when sample-derived linear
phase gradients should be preserved.

``ODTConfig.ewald_embedding_mode`` defaults to
``EwaldEmbeddingMode.TRUNCATE`` for compatibility with legacy Ewald sphere
embedding. Use ``EwaldEmbeddingMode.NEAREST`` for explicit nearest-plane
placement or ``EwaldEmbeddingMode.LINEAR`` to distribute each Ewald sample into
adjacent axial planes with linear weights.

.. automodule:: muscopy.cfg
    :members:

Type Aliases
----------------

.. autodata:: Region
.. autodata:: Regions
.. autodata:: OffsetRegions
.. autodata:: MIPRegion
