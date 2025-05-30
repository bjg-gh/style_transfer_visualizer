# Release Notes

## v1.0.2 — 2025-05-30

**Milestone:** v1.0.2 Patch Maintenance

This patch includes infrastructure and workflow refinements to streamline solo development and reduce CI load.

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

This release formalized the project’s structure and exposed it to the public with full CI and documentation support.

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
