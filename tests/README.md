# Tests

## Shared fixtures

- `make_style_transfer_config`: helper for building `StyleTransferConfig`
 instances with optional overrides. It automatically binds the config to an
 isolated temporary output directory and defaults the device to the current
 `test_device`.
- `style_config_variant`: parametrised version of `make_style_transfer_config`
 that iterates combinations of hardware device and video mode. Use it when
 you want a test to cover multiple runtime layouts without hand-writing the
 permutations.
- `make_input_paths`: builds `InputPaths` objects while letting you override
 individual content/style paths when a test needs bespoke files.
- `make_output_subdir`: resets/creates named subdirectories beneath the shared
 test directory so long-running integration tests can request isolated output
 folders without reimplementing cleanup.
- `output_dir`: exposes the per-test temporary directory as a `Path` for
 cases where raw filesystem access is still the simplest choice.

## Usage tips

- Prefer these fixtures over calling `StyleTransferConfig.model_validate({})`
 directly so tests stay DRY and new defaults propagate automatically.
- When you need to tweak nested sections, pass dictionaries to
 `make_style_transfer_config` (e.g. `video={"save_every": 1}`) and mutate the
 returned config only for per-test specifics.
- Use `make_output_subdir` for cases that previously relied on helper
 functions to manage directories (e.g., in `tests/test_main.py`), and pair it
 with `make_input_paths` when constructing pipeline inputs.
