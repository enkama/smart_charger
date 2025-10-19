# Contributing to smart_charger

Thanks for your interest in contributing! This document explains the PR workflow, how CI approval works for forked PRs, and the branch protection rules maintainers should enable.

## PR workflow (high-level)

1. Fork the repository (if you don't have push access) or create a branch on this repo.
2. Create a feature branch and make small, focused commits.
3. Run the test suite locally:

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

4. Open a pull request (PR) targeting `main` with a clear description and a short changelog entry.

## CI and forked PRs

- Our GitHub Actions workflow `CI` runs on pull requests. For security, workflows from forked PRs are restricted and do not have access to repository secrets.
- If your PR is from a fork and your changes require running workflows that use secrets (for example, publishing steps or external service tokens), a repository maintainer or owner must approve the workflow run:
  - Go to the PR → Actions tab → find the blocked/queued workflow run → click **Approve and run**.
- Maintainers can configure repository settings to require manual approval for first-time contributors. See Repository Settings → Actions → General.

## Branch protection and merge rules (maintainers)

To ensure CI verifies all changes before merging, enable branch protection for `main`:

1. Go to **Settings → Branches → Add rule**.
2. Set the rule for branch name: `main`.
3. Enable **Require status checks to pass before merging**.
4. Add the CI checks to require. The workflow job is named `test` inside the `CI` workflow. On the branch protection UI you should see checks such as:
   - `CI / test (3.11)`
   - `CI / test (3.12)`
   Check the ones you want to require (we recommend requiring both matrix entries for maximum safety).
5. Optionally enable **Require branches to be up to date before merging** to force PRs to re-run CI if `main` changes.
6. Optionally require pull request reviews and limit who can dismiss reviews.

After these rules are enabled, GitHub will block merging until the required checks pass and any review requirements are satisfied.

## Maintainer notes

- If you want PR CI runs from forks to be allowed to access secrets only after a maintainer inspects the code, enable **Require approval for first-time contributors** in Settings → Actions → General.
- For a better developer experience, consider adding a GitHub Action that posts test results or a status summary to the PR comments (optional).

## Need help?

If you're unsure how to proceed, tag a maintainer in the PR or open an issue and request a review.
