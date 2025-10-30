# Release Notes

> To initiate a release, open **New Issue -> Release Checklist**. The form is
> stored at `.github/ISSUE_TEMPLATE/release.yml` and mirrors the steps documented
> in `CONTRIBUTING.md`.

## Unreleased

### Added

- _None yet._

## v1.3.0 - 2025-10-30

**Milestone:** v1.3 UX Notifications & Contributor Docs

This release streamlines contributor onboarding and clarifies model startup
behavior with friendly download logs.

### Highlights

- Published a comprehensive `CONTRIBUTING.md` covering environment setup,
  workflow expectations, labeling, and release steps, now linked from the README
  for quick discovery (#14)
- Added INFO-level logs when pretrained VGG19 weights are downloaded or reused
  so users understand delays on first run (#11)

### Changelog

- Docs: Replaced the draft workflow notes with a polished contributor guide
  detailing tooling commands, branch/commit standards, and release steps (#14)
- UX: Updated VGG initialization to announce whether weights are cached or being
  fetched, with tests enforcing both code paths (#11)

## v1.2.0 - 2025-10-29

**Milestone:** v1.2 Video Polish & Runner Revamp

This release pairs a modernized optimization runner with a polished output pipeline,
delivering richer comparison assets and production-ready video exports.

### Highlights

- Video pipeline now supports post-process encoding, metadata tags, intro
  cards, GIFs, and comparison frames for share-ready outputs (#26, #16, #9,
  #5, #13)
- Introduced `StyleTransferRunner`, shared CLI override helpers, and
  modularized runtime utilities for cleaner orchestration (#56, #57, #55)
- Published reusable compare-grid API and console script backed by decomposed gallery
  modules (#58, #54)

### Changelog

- Video: Added `--video-mode postprocess`, MP4 metadata tagging, optional intro
  title card, GIF output flags, and a finale comparison frame (#26, #16, #5, #9,
  #13)
- Images: Generated a final content/style/result composite image and exposed compare-grid
  helpers with a console script (#10, #58)
- CLI: Moved override logic into the config layer for reuse across entry points (#57)
- Core: Replaced low-level loop helpers with `StyleTransferRunner` that encapsulates
  optimization state (#56)
- Refactor: Split `utils.py` and `image_grid.py` into focused runtime and visualization
  modules (#55, #54)
- Testing: Added shared pytest fixtures for configs and output directories (#59)

## v1.1.0 — 2025-08-27

**Milestone:** v1.1 Core Refactor

This release represents a major internal refactor to improve
maintainability, configuration clarity, and developer workflows,
while preserving full functionality and 100% test coverage.

### Highlights

- Refactored into a clean CLI package with `cli.py` as entry point
- Split monolithic script into multiple focused modules (`core_model`,
  `optimization`, `video`, `image_io`, `utils`, etc.) for clarity and
  maintainability
- Introduced typed configuration system (`config.py`) with TOML support
- Centralized defaults (`config_defaults.py`) and internal constants
- Added CSV loss logging (`loss_logger.py`) with recommended usage for
  long runs
- Strengthened type safety via shared `type_defs.py`
- Improved reproducibility with unified random seed setup
- Reorganized optimization loop with optional disk logging to avoid
  memory bloat
- Centralized logging utilities (`logging_utils.py`) for consistent
  output
- Adopted `uv` for environment and dependency management (replacing
  requirements.txt/venv)
- Completed full linting and typing of the codebase (Ruff + Pyright)
- Streamlined CI with full pre-commit enforcement
- Maintained 100% pytest coverage across all modules

### Changelog

- Core: `main.py` orchestration simplified and tested
- Refactor: replaced single script with modular package structure
- CLI: new argument groups (output, optimization, video, hardware, config)
- Config: `StyleTransferConfig` now mirrors TOML schema directly
- Utils: added stricter input validation and improved error handling
- Docs: clarified configuration, updated README and workflow docs
- CI: added pre-commit hooks and typecheck/lint stages
- Dev: switched to `uv` for dependency resolution and lockfiles
  (simplifies local setup and CI)

## v1.0.2 — 2025-05-30

**Milestone:** v1.0.2 Patch Maintenance

This patch includes infrastructure and workflow refinements to streamline solo
development and reduce CI load.

### Highlights

- Added comprehensive `WORKFLOW.md` with branching and project tracking guidelines
- Skipped CI for documentation-only commits
- Adopted squash merge as default strategy
- Fixed README release badge

### Changelog

- Docs: added effort/priority scale, clarified git practices
- CI: updated `python-ci.yml` with `paths-ignore`
- Maintenance: removed PR clutter from project board

---

## v1.0.1 — 2025-05-28

**Milestone:** First Public Release

This release formalized the project’s structure and exposed it to the public
with full CI and documentation support.

### Highlights

- Added GitHub Actions workflow with Codecov
- Introduced README badges for release and coverage
- Split requirements into core, CUDA, and lockfile variants
- Introduced `__version__.py` for internal version tracking

---

## v1.0.0 — 2025-05-27

**Milestone:** Initial Release

The initial version of the Neural Style Transfer Visualizer.

### Features

- PyTorch-based neural style transfer using VGG19
- Command-line interface with rich configuration options
- Timelapse video output with loss visualization
- Pytest test suite with full (100%) coverage
