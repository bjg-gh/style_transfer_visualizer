---
name: Release
about: Track tasks for cutting a new release.
title: "Release vX.X.X"
labels:
  - release
  - chore
---

<!-- Replace every occurrence of X.X.X with the target version. -->

## Version

- [ ] Confirm release version `X.X.X`

## Tasks

- [ ] Review all issues completed in milestone `vX.X.X`
- [ ] Create a release branch using this issue
- [ ] Update `version` in `pyproject.toml` to `X.X.X`
- [ ] Add release notes to `RELEASES.md`
- [ ] Commit with:

  ```sh
  git commit -m "chore: bump version to X.X.X"
  ```

- [ ] Push and open PR titled `chore: release vX.X.X`
- [ ] Squash and merge PR
- [ ] Tag the release:

  ```sh
  git tag vX.X.X
  git push origin vX.X.X
  ```

- [ ] Create GitHub release:
  - Title `vX.X.X`
  - Description summarizing closed issues or changelog

## Notes

- Link this issue to the release PR with `Closes #...` or mention it manually.
- Delete the release branch after merging.
