Intensity Diffraction Tomography
========================================

``compute_idt`` accepts an optional ``IDTConfig``. By default, IDT uses the
same portable precision contract as ODT: ``float32`` for real arrays and
``complex64`` for complex arrays. Use ``IDTConfig(precision=ArrayPrecision(...))``
to request a different precision.
Use ``IDTConfig(normalization_epsilon=...)`` to configure the small positive
floor used to avoid division by zero during IDT normalization.

The ``z_centering`` setting controls how reconstructed slice indices are mapped
to physical axial positions. ``"central_slice_zero"`` uses the historical
``(idx_z - num_z_slices // 2) * dz`` convention, so slice
``num_z_slices // 2`` is always exactly at ``z = 0``. With an even number of
slices this gives one more negative slice than positive slice. Use this mode
when reproducing older IDT results or when a reconstructed slice must lie
exactly on the focal plane. ``"symmetric_volume"`` uses
``(idx_z - (num_z_slices - 1) / 2) * dz`` and centers the full stack around
``z = 0``. With an even number of slices, zero lies halfway between the two
central slices.

FAQ: Numerical Stability Parameters
-----------------------------------

Why is ``ref_floor_ratio`` ``0.0`` by default?
    ``ref_floor_ratio`` masks reference pixels whose intensity is smaller than
    a fraction of the mean absolute reference intensity. The default is
    ``0.0`` so that IDT does not silently discard dim-but-valid pixels in
    existing workflows. A small absolute floor from ``normalization_epsilon``
    still prevents division by zero. Increase ``ref_floor_ratio`` only when
    dark reference pixels, shadows, or sensor defects create unrealistically
    large normalized contrast values. Values around ``0.01`` to ``0.02`` are a
    practical starting point for noisy reference images.

When should I use ``g_clip``?
    ``g_clip`` limits the normalized intensity contrast after reference
    subtraction and division. The default is ``None`` so IDT does not
    implicitly alter the theoretical normalized contrast. Set a positive value
    only when a few pixels dominate the reconstruction because of dust,
    saturated pixels, registration errors, or defective reference values.

What does ``normalization_epsilon`` protect against?
    ``normalization_epsilon`` is the small absolute floor used where IDT needs
    to divide by a quantity that can become zero or numerically tiny. It
    prevents zero division in intensity normalization and in several inverse
    stability checks. Lower it only when working with well-scaled high-dynamic
    range data where very small but valid values should be preserved.

When should I adjust ``determinant_rel_floor``?
    ``determinant_rel_floor`` stabilizes the coupled real/imaginary inverse
    solve when the transfer functions are nearly linearly dependent or have
    weak support at some spatial frequencies. Increase it if the reconstruction
    contains isolated spikes or non-finite values from poorly conditioned
    frequencies. Decrease it only when you need less regularization and have
    verified that the inverse remains stable.

How do ``check_ifft_imag_residual`` and ``imag_residual_warn_threshold`` help?
    The final inverse FFT should be close to real-valued before IDT takes the
    real part. Enable ``check_ifft_imag_residual`` to warn when the discarded
    imaginary residual is large. Tighten or loosen
    ``imag_residual_warn_threshold`` depending on the expected numerical noise
    for your data and precision setting.

When should I change ``precision``?
    The default portable setting uses 32-bit real arrays and ``complex64``
    arrays. Use higher precision only when JAX x64 support is enabled and you
    have evidence that roundoff error, not model mismatch or input quality, is
    limiting the reconstruction.

When should I change ``z_centering``?
    Use ``"central_slice_zero"`` when reproducing previous results or when one
    reconstructed slice must be exactly at the focal plane. Use
    ``"symmetric_volume"`` when the stack should be geometrically centered
    around ``z = 0``.

.. automodule:: muscopy.idt
    :members:
