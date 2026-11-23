Optical Diffraction Tomography
========================================

`muscopy.odt.ODTParameters` accepts ``img_size_px=(rows, columns)`` as well.
Each cropped spectrum has the shape ``aperturesize_px_xy`` and is embedded on
the Ewald sphere with per-axis lateral frequency pitches. The axial frequency
pitch is the finer lateral pitch, so the reconstruction has the shape
``(*lateral_extent_px_xy, freq_axial_extent_px)`` and reduces to the square
case when both axes are equal.

.. automodule:: muscopy.odt
    :members:
