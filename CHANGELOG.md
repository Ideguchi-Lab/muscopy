# Change log

## Unreleased

### Changed

- Modified lower bound of spectrum on `save_multiangle_spectrum` [[#32](https://github.com/Ideguchi-Lab/microscopy_converters/issues/32)]

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
