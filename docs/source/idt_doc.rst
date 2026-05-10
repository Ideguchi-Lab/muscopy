Intensity Diffraction Tomography
========================================

``compute_idt`` accepts an optional ``IDTConfig``. By default, IDT uses the
same portable precision contract as ODT: ``float32`` for real arrays and
``complex64`` for complex arrays. Use ``IDTConfig(precision=ArrayPrecision(...))``
to request a different precision.

.. automodule:: muscopy.idt
    :members:
