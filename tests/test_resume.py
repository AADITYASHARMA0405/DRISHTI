"""An interrupted run continued with --resume must end where an uninterrupted run ends."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train as train_script  # noqa: E402
from make_toy_data import make  # noqa: E402


def _run(tmp_path, data, root, extra):
    opts = [f"data.json={data.as_posix()}", f"data.image_root={(tmp_path / 'toy' / 'imgs').as_posix()}",
            f"output_root={root.as_posix()}", "train.epochs=4", "noise.clean_epochs=1"]
    return train_script.main(["--config", str(ROOT / "configs" / "smoke_toy.yaml"), "--seed", "3",
                              *extra, "--opts", *opts])


def test_resume_matches_uninterrupted(tmp_path):
    data = make(tmp_path / "toy", persons=40, seed=0)
    full = _run(tmp_path, data, tmp_path / "a", ["--resume"])

    part = _run(tmp_path, data, tmp_path / "b", ["--stop-after", "2"])
    out = Path(part["out_dir"])
    assert part.get("stopped") and (out / "resume.pt").exists() and not (out / ".done").exists()
    rest = _run(tmp_path, data, tmp_path / "b", ["--resume"])

    assert (out / ".done").exists() and not (out / "resume.pt").exists()
    assert [h["epoch"] for h in rest["history"]] == [1, 2, 3, 4]
    for a, b in zip(full["history"], rest["history"]):
        assert abs(a["train_loss"] - b["train_loss"]) <= 0.02 * abs(a["train_loss"]), (a, b)
    assert json.loads((out / ".done").read_text())["epochs"] == 4
