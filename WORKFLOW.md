# Solo Developer GitHub Workflow

A step-by-step workflow for managing issues, branches, commits, and pull
requests as a solo developer using GitHub Projects, squash merges, and milestone tracking.

---

## 1. Create or Choose an Issue

Use the GitHub UI to create an issue for each new task, enhancement, or bug.

- Assign each issue to a **Milestone** (e.g., `v1.3 – UX & Docs`)
- Set the **Effort** (1–5) and **Priority** (High/Medium/Low)
- Use Kanban-style movement:
  - `Backlog` → `Next Up` → `In Progress` → `Done`

<!-- effort-scale -->
### Effort Scale (1-5)
| Effort | Time Estimate         | Description                              |
|--------|-----------------------|------------------------------------------|
| 1      | ~30 min – 1 hour      | Trivial task, minor doc or config change |
| 2      | 1–2 hours             | Small, clear feature or test             |
| 3      | 2–4 hours             | Moderate complexity                      |
| 4      | 4–8 hours             | Large task with multiple steps           |
| 5      | 8–12+ hours           | Very complex, requires coordination or deep refactor |

<!-- priority-criteria -->
### Priority Criteria
| Priority | Meaning                                        | Examples                            |
|----------|------------------------------------------------|-------------------------------------|
| High     | Blocks milestones or core functionality        | Core features, critical refactors   |
| Medium   | Enhances product or user experience            | Useful, not urgent                  |
| Low      | Optional or cosmetic                           | Visual tweaks, nice-to-haves        |

---

## 2. Create a Branch for Each Fix or Feature

### Naming Convention (GitHub Default Style)
Use the format:
```
<issue-number>-<lowercase-dash-separated-title>
```

**Example:**
```
3-docs-add-workflowmd
```

This convention keeps branches traceable and consistent with GitHub’s automatic branch suggestion UI.

### Preferred Method (GitHub UI)
- Navigate to the repo main page
- Open the branch dropdown and accept the default. Edit if you need to.
- Click **Create branch from `main`**

### Alternate Method (Terminal)
```sh
git checkout main
git pull origin main
git checkout -b 3-docs-add-workflowmd
```

---

## 3. Make the Change Locally

Work on your fix or feature locally. When you're ready to commit, use a **short, conventional commit message**.

### Example:
```sh
git add .
git commit -m "docs: fix broken release badge in README"
```

If you're committing only documentation and want to skip CI:
```sh
git commit -m "docs: update README [skip ci]"
```

---

## 4. Push Your Branch

Push your branch to GitHub:
```sh
git push origin <your-branch-name>
```

---

## 5. Open a Pull Request

Once your branch is pushed:

- Open a **pull request**
- Use a **short title** like: `docs: fix broken release badge in README`
- In the PR description, include:
  ```
  Closes #<issue-number>
  ```
- Assign:
  - Yourself as **Assignee**
  - The correct **Labels**
- If you're merging the PR immediately (as is common in solo work), skip
assigning the milestone or project — those should live on the issue.  If not, be sure to also assign:
  - The correct **Milestone**
  - The corresponding **GitHub Project**

---

## 6. Review the Pull Request

Even as a solo dev, use the PR as a checkpoint:
- Review the diff on GitHub
- Ensure test coverage (if applicable)
- Optional: Remove `Co-authored-by:` lines if you're the sole author

---

## 7. Merge the Pull Request

Use **"Squash and Merge"** for a clean, linear commit history.

GitHub will take the PR title and body and turn them into the final commit message. If the PR description includes `Closes #X`, GitHub will automatically close the linked issue after the merge.

There is no need to add `Closes #X` to the squash commit message manually.

---

## 8. Delete the Branch

After merging, delete the feature branch on GitHub (or locally with `git branch -d`).

---

## 9. Confirm the Issue is Closed

GitHub will automatically close the issue when the PR is merged if `Closes #X` was included properly.

Move the issue to **Done** in your GitHub Project board.

---

## 10. Optional: Hide Your Personal Email in Commits

Use GitHub's `noreply` address for privacy:

1. Go to [GitHub → Settings → Emails](https://github.com/settings/emails)
2. Enable `Keep my email addresses private`
3. Set it globally in Git:
   ```sh
   git config --global user.email "YOUR_USERNAME@users.noreply.github.com"
   ```
4. (Optional) Remove local override:
   ```sh
   git config --unset user.email
   ```

---

## 11. Versioning and Releases
After all milestone issues are merged, create an issue for the release
using the template below.  Then follow the steps in the template.

## Release Issue Template
Use the following template to generate an issue for release `vX.X.X`
# Release vX.X.X

This issue tracks the version bump, tagging, and release publishing steps for `vX.X.X`.

## Tasks

- [ ] Review all issues completed in milestone `vX.X.X`
- [ ] Create a release branch:
  ```sh
  git checkout -b release-vX.X.X
  ```
- [ ] Update `__version__.py` to `X.X.X`
- [ ] Commit with:
  ```sh
  git commit -m "chore: bump version to X.X.X"
  ```
- [ ] Push and open PR titled:
  ```
  chore: release vX.X.X
  ```
- [ ] Squash and merge PR
- [ ] Tag the release:
  ```sh
  git tag vX.X.X
  git push origin vX.X.X
  ```
- [ ] Create GitHub Release with:
  - Title: `vX.X.X`
  - Description: Closed issues, changelog, or summary

## Notes
- Link this issue to the release PR using `Closes #XXX` or mention it manually.
- Delete the release branch after merging.

---

**Milestone:** `vX.X.X`  
**Labels:** `release`, `chore`
