Quantitative Phase Imaging
===========================

Rectangular images
++++++++++++++++++++++++++++++++

Pass ``img_size_px=(rows, columns)`` to `muscopy.dh.MuParameters` for a
rectangular hologram. The frequency pitch then differs between the two axes,
so the circular pupil of the objective becomes an ellipse in frequency-pixel
units. `muscopy.dh.get_spectrum` crops that ellipse, and the reconstructed QPI
image has the shape ``MuParameters.aperturesize_px_xy``.

.. code-block:: python

    from muscopy.dh import MuParameters
    from muscopy.qpi import qpi

    params = MuParameters(na=0.8, wavelength_m=500e-9, img_size_px=(1024, 512), px_size_m=5e-6 / 60, n_sol=1.33)
    phase = qpi(hologram, reference, params, offaxis_center)
    assert phase.shape == params.aperturesize_px_xy

Scalar properties such as ``aperturesize_px`` and ``freq_per_px`` are only
defined for square images; use the per-axis ``*_xy`` properties instead.

QPI Calculator
++++++++++++++++++++++++++++++++

.. automodule:: muscopy.qpi
    :members:

QPI Utils
++++++++++++++++++++++++++++++++++++++++++

.. automodule:: muscopy.qpi_utils
    :members:
