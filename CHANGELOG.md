# Changelog #

All notable changes to this project will be documented in this file.

We follow the [Semantic Versioning](https://semver.org/) policy, which uses `MAJOR.MINOR.PATCH` version numbers. In paticular,

- `MAJOR` version when you make incompatible API changes,
- `MINOR` version when you add functionality in a backwards compatible manner, and
- `PATCH` version when you make backwards compatible bug fixes.

## v0.2.0 (TBD) ##

Changes from [v0.1.0](https://github.com/hifirworks/hifir/releases/tag/v0.1.0) to v0.2.0:

- Fixed the issue regarding modifying user-input data in preprocessing calling equilibration.
- Changed I/O to use local buffers instead of shared ones and removed writing to `warn_flag` at the beginning of factorizations.
- Made logging system more complete.
- Added C library `libhifir`.
- Updated documentation for the C interface.
- Added MatrixMarket file readers for the C library.
- Added example code for `libhifir`.

[Full Changelog](https://github.com/hifirworks/hifir/compare/v0.1.0...HEAD)

## [v0.1.0](https://github.com/hifirworks/hifir/releases/tag/v0.1.0) (2021-08-22) ##

This is the initial official release of the HIFIR package.
