# ACT Backend Evaluation Repository

This repository contains compiler backends for the hypothetical accelerators `QKV` and `QKV_new`, generated with both the standard ACT backend generator and the improved-extraction ACT backend generator.

## Repository Structure

- `act-targets/`
  - Contains accelerator target definitions and backend code used by the generator pipeline.
- `scripts/`
  - Provides helper scripts for building, compiling, and testing the generated backends.
- `tests/`
  - Includes automated tests to validate backend correctness and compare outputs.

## Using the Tests Repository

The `tests/` folder is designed to help verify the correctness of generated accelerator backends and compare outputs across backend variants.

### Run the tests

From the repository root:

```bash
./tests/pytest.sh
```

This script runs the available backend validation tests for `QKV`, `QKV_new`, `QKV_improved`, and `QKV_new_improved`.

### What the tests do

- Build or load the relevant accelerator backend.
- Execute the backend on reference input data.
- Compare the computed outputs with expected results.
- Report mismatches or failures so the generated backend behavior can be validated.

## Using the Scripts Repository

The `scripts/` folder contains utilities to build and test accelerator compiler backends and to generate assembly kernels from HLO descriptions.

### Build and generate backends

Use the helper scripts to compile the HLO, generate a backend, and produce assembly kernel outputs.

```bash
cd /home/nitink/cs526/act-smoothe-evaluation/scripts
./build_backend.sh
./compile_hlo.sh
./generate.sh
```

### Generate assembly kernels

The assembly output is written into the repository under the `asm/` directory. Each script performs a step in the generation pipeline:

- `generate.sh` — generates compiler backend code for the accelerator, along with the backend binary.
- `build_backend.sh` — builds backend binary from compiler backend code (this step is usually covered if generate.sh runs properly)
- `compile_hlo.sh` — compiles the HLO kernel descriptions into an accelerator kernel.

## Comparing Backend Outputs

This repository is structured to make it easy to compare the standard ACT outputs against the improved extraction variant. Use the tests and generated assembly artifacts together to ensure the backends are valid and behave consistently.

## Notes

- Use `tests/pytest.sh` for validation and regression checking.
- Use the scripts in `scripts/` when you want to generate or regenerate assembly kernels from the accelerator backends.
- Review `asm/` for generated kernel examples and verify them against expectations.
