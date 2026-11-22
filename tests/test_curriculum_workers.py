"""Regression test: the noise ratio set by set_epoch() must reach DataLoader workers, including persistent
workers that are started once and reused for every epoch."""
import sys
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_toy_data import make  # noqa: E402

from drishti.config import load_config  # noqa: E402
from drishti.data import PedesTrain, collate_train  # noqa: E402


def _fraction(dl):
    ks = [b["k"] for b in dl]
    return sum(int((k > 0).sum()) for k in ks) / sum(len(k) for k in ks)


@pytest.mark.parametrize("persistent", [False, True])
def test_noise_ratio_reaches_workers(tmp_path, persistent):
    data = make(tmp_path / "toy", persons=20, seed=0)
    cfg = load_config(ROOT / "configs" / "smoke_toy.yaml",
                      [f"data.json={data.as_posix()}", f"data.image_root={(tmp_path / 'toy' / 'imgs').as_posix()}",
                       "masking.prob=0"])
    ds = PedesTrain(cfg, seed=0)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_train,
                    persistent_workers=persistent)
    ds.set_epoch(0, 0.0)
    assert _fraction(dl) == 0.0
    ds.set_epoch(1, 1.0)
    assert _fraction(dl) > 0.95
