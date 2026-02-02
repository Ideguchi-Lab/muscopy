# Change log

## Unreleased

---

## Version 0.7.3 [2026-02-02]

### Added

- Linear approximation mode for ODT reconstruction [[#117](https://github.com/Ideguchi-Lab/muscopy/pull/117)]
  - Added `linear_approx` option to `ODTConfig` for linearized scattering potential to refractive index conversion
  - New `calc_scattering_potential()` function extracted from `odt()` for direct access to scattering potential and spectrum

### Removed

- Removed `pt_signal_1st_order()` function from `odt` module [[#117](https://github.com/Ideguchi-Lab/muscopy/pull/117)]

---

## Version 0.7.2 [2025-12-15]

### Added

- `use_skimage` option in `unwrap_phase()` function [[#109](https://github.com/Ideguchi-Lab/muscopy/issues/109)]

### Fixed

- Fixed phase unwrap algorithm instability [[#109](https://github.com/Ideguchi-Lab/muscopy/issues/109)]
  - Corrected divergence (rho) calculation in `unwrap_phase()` to use proper Neumann boundary conditions
  - Changed from `prepend=0.0` to `prepend=0.0, append=0.0` for correct DCT-based Poisson solver compatibility
  - Fixed ROI handling to only use gradients where both endpoints are inside ROI, preventing boundary artifacts
  - Updated `keep_mean` to use ROI-masked mean when ROI is specified
  - Added comprehensive test suite for `unwrap_phase()` function

---

## Version 0.7.1 [2025-12-11]

### Added

- Phasor analysis for multi-dimensional spectral imaging data [[#113](https://github.com/Ideguchi-Lab/muscopy/issues/113)]
  - `PhasorResult`: Named tuple to store phasor analysis results (g, s, i_sum)
  - `PhasorParameters`: Configuration class for phasor analysis with wavenumbers validation
  - `phasor()`: Core function to calculate phasor components from 2D or 3D spectral imaging data
  - Supports automatic detection of spatial dimensionality (3D array → 2D spatial, 4D array → 3D spatial)
  - Configurable spectral axis position via `spectral_axis` parameter
  - Comprehensive test coverage with 21 test cases

### Fixed

- Fixed edge_size parameter handling in ODT reconstruction
  - Updated `_shift_dh_spectrum()` function to properly accept and use `edge_size` parameter
  - Fixed `synthesize_spectrum()` to remove incorrect edge_size subtraction from spectrum dimensions
  - Updated `odt()` function to pass `edge_size` parameter from config to spectrum shifting operations
  - Added proper TODO comment with author and issue reference for future FFT factor corrections
  - Resolved ruff linting errors related to TODO comment formatting
- Shift z coord of `odt.pt_signal_1st_order` to center
  - Added `jnp.fft.fftshift()` to `pt_signal` output in `pt_signal_1st_order()` function
  - Ensures z-axis centering of PT signal for consistent visualization and analysis
  - Maintains compatibility with existing workflows by only modifying output centering

---

## Version 0.7.0 [2025-08-13]

### Added

- Gradient correction in ODT
- Enhanced aberration correction API with pupil function support
  - Added optional `pupil_func` parameter to `offaxis_dh()` function for direct aberration correction during reconstruction
  - Added optional `pupil_func` parameter to `qpi()` function for streamlined QPI phase calculation with aberration correction
  - Enables seamless integration of aberration correction into existing workflows without manual spectrum processing
  - Maintains backward compatibility with existing code that doesn't use aberration correction
- Optical aberration correction functionality
  - Added `correct_aberration()` function in `dh` module for correcting optical aberrations using pupil functions
  - Supports correction of spherical aberration, astigmatism, coma and other wavefront distortions
  - Enables restoration of image quality degraded by optical system imperfections
- Comprehensive aberration correction demonstration
  - Added `examples/qpi_aberration_correction.py` with complete workflow demonstration
  - Shows aberration simulation, QPI reconstruction with aberration, and correction process
  - Includes detailed visualization comparing without aberration, with aberration, and corrected results
  - Provides quantitative analysis with correction efficiency, SNR improvement, and error metrics
  - Demonstrates 83.3% correction efficiency and 15.6 dB SNR improvement in example case
  - Updated to use simplified API with `qpi()` function for cleaner, more readable code

### Fixed

- Fixed Sphinx documentation build warnings and improved CI compatibility
  - Updated example files to use proper Sphinx-Gallery format with titles and cell delimiters
  - Added environment-based ignore pattern for MLB simulation examples in CI environments
  - Configured conditional file inclusion based on availability of muscopy_mlbsim dependency
  - Resolved all toctree warnings, missing title warnings, and cross-reference issues
  - Enabled warning-free documentation builds in both local development and GitHub Actions environments
- Improved example file robustness and user experience
  - Added muscopy_mlbsim availability check to `idt_with_mlb_simulation.py` with graceful fallback
  - Enhanced `visualize_transfer_functions.py` to show plots inline by default while maintaining file save option
  - Improved error handling for missing dependencies in example scripts

---

### Version 0.6.3 [2025-08-12]

### Added

- Regularization term to suppress ODT reconstruction error.

### Fixed

- Fixed critical calculation errors in ODT reconstruction
  - Corrected index calculations in `_embed_3d_spectrum()` to use proper range bounds
  - Fixed scattering potential calculation using `k_per_px` instead of incorrect `freq_per_px`
  - Fixed illumination vector sign errors in spectrum shifting and mask calculations
  - Improved numerical stability by removing unnecessary epsilon additions in logarithmic calculations
  - Fixed arange usage for proper coordinate grid generation
  - Updated example to use correct off-axis center coordinates for accurate spectrum extraction

---

## Version 0.6.2 [2025-08-09]

### Added

- Edge removal functionality in ODT first-order scattering wave calculation [[#98](https://github.com/Ideguchi-Lab/muscopy/issues/98)]
  - Added `edge_size` parameter to `_calc_1st_scattering_spectrum()` function
  - Removes edge pixels from complex phase fields to minimize boundary artifacts
  - Configurable edge removal size for optimal reconstruction accuracy
- Phase gradient correction functionality [[#47](https://github.com/Ideguchi-Lab/muscopy/issues/47)]
  - Added `correct_gradient()` function in `dh` module for removing linear phase gradients
  - Supports `edge_size` parameter to exclude edge pixels from gradient estimation
  - Uses median-based gradient estimation for robustness against outliers
  - Handles phase wrapping issues through complex exponential unwrapping
  - Separates gradient correction from constant phase offset correction for modularity
- Phase gradient correction demonstration
  - Added `examples/phase_gradient_correction.py` with comprehensive visualization
  - Shows step-by-step correction process (gradient removal + offset correction)
  - Includes quantitative analysis with proper phase unwrapping
  - Demonstrates edge exclusion functionality

### Changed

- Fixed illumination shift calculation in `get_spectrum()` method
  - Changed illumination shift calculation from using `shape[0] // 2, shape[1] // 2` to using `offaxis_center[0], offaxis_center[1]`
  - Provides more accurate illumination angle tracking by using the actual off-axis center instead of assuming centered coordinates
  - Improves precision in IDT reconstruction workflows where illumination angle accuracy is critical
- Corrected scaling factor in ODT reconstruction
  - Fixed scaling factor from `(2π)³` to `(2π)^(3/2)` in both `odt()` and `pt_signal_1st_order()` functions
  - Ensures proper physical units and normalization in refractive index reconstruction
- Optimized type conversions in ODT calculations
  - Replaced `astype()` calls with `jnp.asarray()` for more efficient array type handling
  - Added proper type specifications to improve numerical stability

---

## Version 0.6.1 [2025-07-31]

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
- Extended `print_illumination_angle` option to `get_spectrums()` method [[#96](https://github.com/Ideguchi-Lab/muscopy/issues/96)]
  - Adds same illumination angle analysis capability for multiple spectra processing
  - Displays spectrum index for each calculation when enabled
  - Provides consistent functionality across single and multiple spectrum workflows

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
