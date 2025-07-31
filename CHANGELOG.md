# Change log

## Unreleased

### Added

- `print_illumination_angle` option in `get_spectrum()` method [[#94](https://github.com/Ideguchi-Lab/muscopy/issues/94)]
  - Adds ability to print illumination angle in NUMPY coordinate system
  - Calculates angle from maximum value coordinate as [max_coord[0] - shape[0]//2, max_coord[1] - shape[1]//2]
  - Useful for IDT reconstruction where illumination angle tracking is necessary
- Illumination NA calculation in `get_spectrum()` method [[#96](https://github.com/Ideguchi-Lab/muscopy/issues/96)]
  - Calculates and prints illumination NA when `print_illumination_angle` is True
  - Uses formula: illumination_na = |illumination_shift|/(params.aperturesize_px//2) * params.na
  - Provides quantitative measure of illumination numerical aperture
  - Also calculates and prints theta angle from illumination_shift = [cos(theta), sin(theta)]
  - Displays theta in both radians and degrees for convenient analysis

---

## Version 0.6.0 [2025-07-29]

### Added

- Intensity Diffraction Tomography (IDT) support [[#56](https://github.com/Ideguchi-Lab/muscopy/issues/56)]
  - `compute_idt()`: Core IDT reconstruction functionality
  - `IDTParameters`: Configuration class for IDT parameters
  - Green's function computation for IDT physics
  - Transfer function calculation and visualization
  - Support for conversion to refractive index maps
  - Integration with MLB simulation for IDT examples
  - Memory management and debugging tools for IDT computation
- Enhanced debugging and visualization capabilities
  - Transfer function visualization debug tool
  - Slice visualizer option for 3D data inspection
  - Intensity image save/load functionality
  - Debug prints for NaN identification in computations

### Changed

- specify version requirements for documentation dependencies [[#67](https://github.com/Ideguchi-Lab/muscopy/issues/67)]
- Improved pupil function consistency with Green's function
- Enhanced memory management to prevent accumulation issues
- Updated illumination angle algorithms for better accuracy
- Normalized transfer functions to prevent NaN issues
- Fixed coordinate transformation in Green's function computations

### Fixed

- IDT reconstruction issues and improved debugging tools [[#92](https://github.com/Ideguchi-Lab/pull/92)]
- NaN issues in IDT computation by normalizing transfer functions
- Shape mismatch errors in IDT with MLB simulation
- Memory accumulation issue in IDT computations
- Fourier transform pipeline consistency in IDT
- Missing inverse Fourier transform in IDT reconstruction
- Incoherent mask handling
- Scaling factors in transfer function computations

---

## Version 0.5.2 [2025-07-10]

### Added

- ODT difference calculation functionality [[#48](https://github.com/Ideguchi-Lab/muscopy/issues/48)]
  - `calculate_odt_difference()`: Calculate the difference between two ODT reconstructions with same parameters
  - `pt_signal_1st_order()`: Calculate Photothermal signal with 1st order approximation
  - Added comprehensive test coverage for input validation
- ODT demonstration with `muscopy-mlbsim` library.

---

## Version 0.5.1 [2025-06-30]

### Added

- Inline digital holography reconstruction functionality [[#57](https://github.com/Ideguchi-Lab/muscopy/issues/57)]
  - `ps_idh_reconstruct()`: Core phase-shifting inline digital holography reconstruction
  - `inline_dh()`: High-level interface for inline DH with validation
  - Support for known phase shifts with future blind reconstruction capability
  - Addresses sampling resolution limitations of off-axis DH beyond Nyquist-Shannon limit

### Changed

- Improved `unwarp_phase` routine to avoid unintended modification [[#73](https://github.com/Ideguchi-Lab/muscopy/issues/73)]

---

## Version 0.5.0 [2025-06-03]

### Added

- typecheck CI [[#62](https://github.com/Ideguchi-Lab/muscopy/issues/62)]

### Changed

- Changed the array backend from numpy/cupy to JAX [[#53](https://github.com/Ideguchi-Lab/muscopy/issues/53)]
  - This improved type stability [[[#62](https://github.com/Ideguchi-Lab/muscopy/issues/62)]]

### Fixed

- Fix bugs in ODT procedures

### Removed

- `backend_manager` module [[#53](https://github.com/Ideguchi-Lab/muscopy/issues/53)]
- Removed support for `py39`, `py310`.
- Removed support for `macOS`.

---

## Version 0.4.2 [2025-05-15]

### Added

- Phase correct function dedicated for QPI and MIP-QPI [[#66](https://github.com/Ideguchi-Lab/muscopy/issues/66)]

### Fixed

- Bug in the non-recursive directory parser methods [[#65](https://github.com/Ideguchi-Lab/muscopy/issues/65)]

---

## Version 0.4.1 [2025-05-13]

### Added

- Set up sphinx document and related CI [[#49](https://github.com/Ideguchi-Lab/muscopy/issues/49)]

---

## Version 0.4.0 [2025-05-13]

### Added

- Demo of QPI reconstruction [[#30](https://github.com/Ideguchi-Lab/muscopy/issues/30)]
- New module dedicated for optical diffraction tomography `odt.py`.
- `offaxis_dh` method in `qpi` module.
- demultiplexing method in `dh` module. [[#51](https://github.com/Ideguchi-Lab/muscopy/issues/51)]
  - Added an example in `examples/multiplexed_qpi.py`
- ruff check for every PR.

### Changed

- QPI module now explicitly requires `backend` parameter [[#30](https://github.com/Ideguchi-Lab/muscopy/issues/30)]
  - Especially, `offaxis_center` is removed from `QPIParameters` class for the support of multiplexed QPI.
- Updated `shot_noise_calculation.py` for the latest version.
- Modified `unwrap_phase` backend.
- Changed module name from `unwrap_phase` to `qpi_utils`.
- Moved digital holography modules from `qpi` to `dh` [[#54](https://github.com/Ideguchi-Lab/muscopy/issues/54)]
- Use overload for `qpi.qpi` and `dh.offaxis_dh`

### Removed

- `fft2d` module
- `aperture_synthesis` module

---

## Version 0.3.1 [2025-04-08]

### Added

- Introduced Ruff as a project formatter and linter [[#38](https://github.com/Ideguchi-Lab/muscopy/issues/38)]
- pre-commit config file
- `backend_manager` module to control CPU/GPU [[#41](https://github.com/Ideguchi-Lab/muscopy/issues/41)]
- setup pytest workflow

### Removed

- Removed black, isort, and flake8.

---

## Version 0.3.0 [2025-04-06]

### Added

- Supported backward scattering wave for ODT reconstruction [[#42](https://github.com/Ideguchi-Lab/muscopy/issues/42)]

---

## Version 0.2.1 [2025-03-20]

### Changed

- Added sensor noise factor in phase calculation [[#37](https://github.com/Ideguchi-Lab/microscopy_converters/issues/37)].

### Fixed

- Fix release workflow.

---

## Version 0.2.0 [2025-02-16]

### Changed

- Modified lower bound of spectrum on `save_multiangle_spectrum` [[#32](https://github.com/Ideguchi-Lab/microscopy_converters/issues/32)]
- Made phase noise calculator to use photon number instead of electron[[#35](https://github.com/Ideguchi-Lab/microscopy_converters/issues/35)]

### Fixed

- No-precision errors in `aperture_synthesis.py`[[#34](https://github.com/Ideguchi-Lab/microscopy_converters/pull/34)]

---

## Version 0.1.5 [2024-10-13]

### Added

- Created dtype manager([#16](https://github.com/Ideguchi-Lab/microscopy_converters/issues/16))
- Created `MUSCOPY_GPU` environment to switch CPU/GPU([#18](https://github.com/Ideguchi-Lab/microscopy_converters/issues/18))
- Created pixel-wise shot noise calculation([[#5](https://github.com/Ideguchi-Lab/microscopy_converters/issues/5)])
- Created automatic release workflow

### Fixed

- Adapt high-NA condition in ODT([[#21](https://github.com/Ideguchi-Lab/microscopy_converters/issues/21)])

### Changed

- Refactor QPI/ODT parameters with dataclass ([#23](https://github.com/Ideguchi-Lab/microscopy_converters/issues/23))

---

## Version 0.1.4 [2024-05-08]

### Fixed

- $2\pi$ factor in ODT
- Fixed iterative ODT reconstruction

### Removed

- Removed ODT demo because it will be hard to maintain

---

## Version 0.1.3 [2024-04-16]

### Added

- Compressor for numpy arrays in `muscopy.compressor.py`
- Ancillary methods for the above
- Create options for loading compressed pickle data in aperture synthesis

### Fixed

- Import error in `phase_unwrap` module when importing in GPU less env

---

## Version 0.1.2 [2024-04-07]

### Added

- Phase unwrapping func created by Horie-san
- Options for `correct_offset` to choose whether correct phase and amplitude offsets.

### Fixed

- Add phase unwrapping func into Rytov approximation so that the ODT synthesizer is able to reconstruct refractive index map successfully

---

## Version 0.1.1 [2024-04-01]

First release
