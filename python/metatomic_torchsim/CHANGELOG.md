# Changelog

All notable changes to metatomic-torchsim will be documented in this file.

<!-- towncrier release notes start -->

## metatomic-torchsim v0.1.0 (2026-03-02)

### Added

- Initial release of ``metatomic-torchsim`` with ``MetatomicModel`` wrapper
  adapting metatomic models to TorchSim's ``ModelInterface`` protocol.
  Supports batched simulations, output variants, uncertainty quantification,
  non-conservative forces/stress, and additional model outputs. ([#167](https://github.com/metatensor/metatomic/issues/167))
