# Publishing checklist (human steps)

The step-by-step list to follow is [`RELEASE_CHECKLIST.md`](../RELEASE_CHECKLIST.md). It also has the announcement plan and the draft upstream bug reports; this page gives background on each step.

Nothing is pushed or published from the development machine. The repository, the PyPI project and the Pages site are all created by a maintainer, by hand, following the steps below. Do them in order.

## 1. Before the first push

- [ ] Check that the names are still free:
  - PyPI: `https://pypi.org/pypi/canitoolcall/json` returns 404.
  - GitHub: the repository `redd34/canitoolcall` does not exist yet (`https://api.github.com/repos/redd34/canitoolcall` returns 404).
- [ ] Decide which identity publishes the project. Every commit so far is authored with the repository-local `git config user.email`; if that should be a personal address, set it and rewrite the (still local) history before the first push, for example `git rebase -r --root --exec 'git commit --amend --no-edit --reset-author'`. Check whether an employer agreement covers the project.
- [x] Choose the GitHub owner: the repository is `https://github.com/redd34/canitoolcall` (the personal account `redd34`, no org). The owner is filled in everywhere: the clone and matrix URLs in `README.md`, the `fancy-pypi-readme` substitution in `pyproject.toml` (it makes the README's relative links absolute on PyPI), `[project.urls]` (`Homepage`, `Issues`, `Changelog`, `Matrix`) and the private vulnerability reporting link in `SECURITY.md`.
- [ ] Replace the contact method in `CODE_OF_CONDUCT.md` (search for `CONTACT METHOD`).
  - `python scripts/check_release.py` must then report no placeholders; the release workflow runs it too.
- [ ] Run the full local check, and confirm the build is clean:

  ```sh
  uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run mypy && uv run pytest
  uv run canitoolcall validate
  uv build && uvx twine check --strict dist/*
  ```

- [ ] Audit the fixtures before release. Every fixture must have provenance. Re-run every `scripts/fixtures/*/build.py`, and check that `git diff` stays clean.
- [ ] Make sure only dated snapshots under `results/YYYY-MM-DD/` are committed (other results files and `/site/_build/` are gitignored), and that `.engines/` and `.venvs/` are ignored too.

## 2. GitHub repository

- [ ] Create the repository (public) and push `main`.
- [ ] Under **Settings → Pages**, set **Source** to **GitHub Actions**.
- [ ] Under **Settings → Code security**, enable **Private vulnerability reporting**, which `SECURITY.md` relies on.
- [ ] Under **Settings → Environments**:
  - The `github-pages` environment is created automatically on the first deploy.
  - Create an environment named `pypi`, and add required reviewers so a release needs an approval.
- [ ] Protect `main`: require the `ci` checks to pass before merging.
- [ ] Run **Actions → nightly → Run workflow** once by hand. Check that:
  - each engine job either uploads a results file or shows a "not run" notice;
  - the site builds;
  - the Pages URL works.

  Then put that URL into `README.md` in place of the matrix link placeholder.

## 3. PyPI (trusted publishing, no API token)

- [ ] On PyPI, go to **Your account → Publishing → Add a new pending publisher** and fill in:

  | Field | Value |
  |---|---|
  | PyPI project name | `canitoolcall` |
  | Owner | the GitHub org or user |
  | Repository | `canitoolcall` |
  | Workflow | `release.yml` |
  | Environment | `pypi` |

- [ ] Set `__version__` in `src/canitoolcall/__init__.py`, for example `0.1.0`, and the `version` input's default in `action.yml` to the same value (a test checks they match), and commit.
- [ ] Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
- [ ] Go to **Actions → release → Run workflow**, choose the tag `v0.1.0` in "Use workflow from", and approve the `pypi` environment. The workflow refuses to run on anything but a `v*` tag, and fails if the tag does not match the built version.
- [ ] Check the release from a clean machine: `uvx canitoolcall --version` and `uvx canitoolcall probe --help`.
- [ ] Move the GitHub Action's major tag to the release so `uses: redd34/canitoolcall@v0` picks it up: `git tag -f v0 v0.1.0 && git push -f origin v0` (only after PyPI has the release, because the action installs it from there).

## 4. After release

- [ ] Announce the matrix to engine maintainers, and offer the pytest plugin for their CI (see the README).
- [ ] Before filing any upstream issue, turn the untriaged candidates in `docs/DESIGN.md` §6 into fixtures and verify them.
