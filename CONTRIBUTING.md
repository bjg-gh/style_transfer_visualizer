# Contributing

Style Transfer Visualizer runs with the expectation of a professional solo
workflow: every change starts from an issue, follows a consistent branch and PR
process, and keeps quality gates (formatting, type checking, coverage) at 100%.
This guide captures the path so collaborators can plug in without surprises.

---

## Quick Start

### Environment setup

```bash
uv venv --python=3.12
uv sync --extra dev --extra cu128  # use --extra cpu on machines without CUDA
```

If you want to activate the virtual environment before running commands:

- PowerShell: `.venv\Scripts\Activate.ps1`
- bash/zsh/fish: `source .venv/bin/activate`

### Install pre-commit hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files  # confirm a clean baseline
```

Hooks cover Ruff (linting, formatting, import sorting), general hygiene checks,
and Pyright. They must succeed before every commit.

### Handy commands

```bash
uv run style-visualizer --help              # inspect CLI usage
uv run compare-grid --help                  # gallery CLI quick reference
uv run pytest                               # full suite, 100% coverage required
uv run pytest -m "not slow"                 # skip slow markers
uv run pytest -m "visual" --maxfail=1 -vv   # run only visual snapshot tests
uv run pyright                              # type checking
uv run ruff check                           # lint only (pre-commit runs --fix)
```

Coverage HTML is emitted to `htmlcov/index.html`; open it in a browser whenever
you need a visual gap check.

---

## Workflow Expectations

### Issues and planning

- Open an issue for every task, even for solo work.
- Move cards left-to-right on the project board:
  `Backlog` -> `Next Up` -> `In Progress` -> `Done`.
- Attach milestones so release burndown stays accurate.
- Record an **Effort** score (1-5):

| Effort | Time estimate       | Typical work                                   |
| ------ | ------------------- | ---------------------------------------------- |
| 1      | 30 min - 1 hour     | Minor docs/config tweak, tiny bug fix          |
| 2      | 1 - 2 hours         | Small feature or focused test additions        |
| 3      | 2 - 4 hours         | Moderate feature or partial refactor           |
| 4      | 4 - 8 hours         | Multi-step feature touching several modules    |
| 5      | 8 - 12+ hours       | Deep refactor or cross-cutting architectural work |

- Use priority to track urgency:

| Priority | Meaning                               | Examples                                   |
| -------- | ------------------------------------- | ------------------------------------------ |
| High     | Blocks release or core workflows      | Broken CLI, missing asset, regression      |
| Medium   | Improves UX or dev ergonomics         | New flag, doc rewrite, tooling fix         |
| Low      | Cosmetic or aspirational              | Visual polish, optional enhancements       |

Document acceptance criteria and manual verification steps directly on the
issue so the PR description can stay lean.

### Branches

- Branch from `main` for every issue.
- Name branches `<issue-number>-<lowercase-dash-title>` (for example,
  `14-docs-contributing`).
- Push frequently so GitHub links the branch back to its issue automatically.

### Commits

- Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`).
- Small, descriptive commits make squash merges easier to review later.
- Append `[skip ci]` only when the change is documentation-only and you are
  certain lint/tests/type checks are unaffected.

### Pull requests

- PR title should match the squash commit you expect to land (`docs:`, `feat:`,
  etc.).
- Include `Closes #<issue-number>` in the description plus any manual checks you
  performed.
- Assign yourself and add labels before merging.
- Confirm the quality gates (pre-commit, pytest, Pyright) are green.
- Merge with squash to keep history tidy, then delete the branch and move the
  issue to `Done`.

### Labels

Labels drive triage, automation, and reporting. Apply at least one from each
category below:

- **Type**: `bug`, `enhancement`, `documentation`, `test`, `chore`, `refactor`,
  `release`.
- **Area**: `cli`, `video`, `image-processing`, `metadata`, `training`,
  `configuration`, `ci`, `UX`, etc.
- **State**: `help wanted`, `good first issue`, `question`, `performance`,
  `maintainability`.

Pick the most specific label available. Add a comment on the issue if the scope
changes so the label can follow.

---

## Coding Standards

- **Formatting**: Ruff controls lint and format (`line-length = 79`,
  docstring code blocks wrap at 72). Rely on pre-commit to keep files compliant.
- **Type hints**: Required for all new Python code. Prefer standard library
  types from `typing`/`collections.abc`. Run `uv run pyright` to verify.
- **Imports**: Keep sorted by Ruff. Avoid hand-tuning order.
- **Docstrings**: Describe what the function does and any constraints a test
  cannot encode. Reference upstream documentation sparingly in comments.
- **DRY**: Extract helpers when behavior repeats. Keep modules cohesive.
- **Errors**: Raise explicit exceptions for invalid states; avoid silent
  failures that hide bugs.

---

## Testing & Quality Gates

- `uv run pytest` must pass locally before every PR. Coverage is enforced at
  100% with branch coverage enabled (`--cov-branch`).
- Mark long-running or GPU-heavy tests with `@pytest.mark.slow` so you can run
  a targeted subset with `-m "not slow"`.
- Mark tests that rely on rendering or visual output with `@pytest.mark.visual`.
- Use `@pytest.mark.integration` for tests that depend on external binaries,
  large fixtures, or network access.
- Snapshot new CLI text output so regressions show up automatically.
- Seed randomness (`torch.manual_seed`, `random.seed`) whenever determinism is
  required for tests or image generation.

Helpful pytest flags while iterating:

```bash
uv run pytest --lf             # rerun only the previous failures
uv run pytest -k "cli"         # run tests whose names contain 'cli'
uv run pytest -m "slow"        # run slow suite explicitly
```

Review `coverage.xml` or `htmlcov/index.html` to ensure no lines are missed
before opening the PR.

---

## Manual CLI Verification

Automated tests catch most regressions, but the CLI is worth spot checking for
major features or releases:

```bash
uv run style-visualizer \
  --content tests/fixtures/content.jpg \
  --style tests/fixtures/style.jpg \
  --steps 25 --save-every 5 \
  --final-frame-compare
```

```bash
uv run compare-grid \
  --content tests/fixtures/content.jpg \
  --style tests/fixtures/style.jpg \
  --layout gallery-stacked-left
```

Store example outputs in `mock_output/` if you plan to reference them in issues
or release notes.

---

## Pre-commit Details

The project ships with a curated `.pre-commit-config.yaml`. Day-to-day flow:

1. `uv run pre-commit install` (run once per machine).
2. Hooks run automatically on `git commit`; fix and re-stage any reported
   issues.
3. To refresh hook revisions, open an issue labeled `dependencies` or `chore`,
   bump versions, and run the release checklist.

Avoid committing with `--no-verify`; CI runs the same hooks and will fail.

---

## Releases & Versioning

- Versions live in `pyproject.toml` using `MAJOR.MINOR.PATCH.devN`.
- Increase `.devN` with each merged issue inside a milestone.
- When it is time to cut a release, open **New Issue -> Release Checklist**. The
  form (`.github/ISSUE_TEMPLATE/release.yml`) pre-populates the full task list
  so nothing is skipped.
- Follow `RELEASES.md` alongside the issue form: bump the version, update
  release notes, tag (`vX.Y.Z`), and publish.
- Immediately after tagging, bump `pyproject.toml` to the next `.dev0` so `main`
  reflects active development.

---

## Additional References

- Day-to-day usage and CLI examples live in `README.md`.
- Release history, templates, and migration notes are documented in
  `RELEASES.md`.
- If something feels unclear, start a conversation in the related issue before
  shipping code.

Thanks for helping keep the project clean, consistent, and fully covered.
