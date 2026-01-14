# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-01-13

### Changed

- **BREAKING**: Minimum Python version is now 3.9 (dropped 3.8 support)
- Updated all dependencies to modern versions:
  - `numpy>=1.24` (removed `<1.24` constraint)
  - `lightning>=2.0.0` (PyTorch Lightning 2.x)
  - `mir_eval>=0.8.0` (NumPy 2.x compatible)
  - `natten>=0.17.0` (now required for all platforms, not just macOS)
  - `librosa>=0.10.0`
  - `demucs>=4.0.0`
  - `timm>=0.9.0`
  - And other dependencies updated to latest stable versions
- Fixed deprecated Lightning API usage:
  - Replaced `trainer.checkpoint_callback` with explicit callback lookup
  - Updated `load_from_checkpoint` usage to class method pattern
- Fixed deprecated PyTorch pattern: replaced `tensor.data.fill_()` with `torch.no_grad()` context

### Added

- Modal cloud training support (`modal_train.py`) for GPU training on Modal infrastructure
- Python 3.12 support

### Removed

- Python 3.8 support
- PyPy support (not commonly used for ML workloads)
- macOS-only NATTEN conditional (now required on all platforms)

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/mir-aidj/all-in-one/compare/v1.0.3...v1.1.0
