# Team workflow

## Branches and reviews
* `main` must always pass `python -m pytest -q`.
* One branch per task: `<name>/<short-task>`, e.g. `dia/noise-indian-confusions`.
* Open a pull request; **another member reviews it** before merging. Reviewers run the tests and read the
  diff. This is how each of us learns the whole system.
* Small, focused pull requests (under ~300 changed lines) get reviewed faster.

## Experiments
* Every experiment is a config file in `configs/` (copy and edit; don't hard-code values in Python).
* Name runs `<phase>_<idea>` (e.g. `p3_crossling_l2`); seeds are appended automatically.
* Log runs to Weights & Biases (`wandb.enabled: true`) and note the git commit in the run.
* Never commit datasets, checkpoints or `outputs/` (see `.gitignore`).

## Weekly rhythm
* **Monday sync (30 min):** show results, not plans. Each blocker gets an owner.
* **Fortnightly guide meeting:** bring the latest `outputs/summary.md` and one decision to ask for.

## Definition of done
Merged after review · runs from a config · results logged with the commit · README updated if usage changed.
