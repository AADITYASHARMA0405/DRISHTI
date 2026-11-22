"""Regression test: the noise ratio set by set_epoch() must reach DataLoader worker processes."""
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_toy_data import make  # noqa: E402

from drishti.config import load_config  # noqa: E402
from drishti.data import PedesTrain, collate_train  # noqa: E402


def _corrupted_fraction(ds, epoch, ratio):
    ds.set_epoch(epoch, ratio)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_train,
                    persistent_workers=False)
    ks = [b["k"] for b in dl]
    total = sum(len(k) for k in ks)
    return sum(int((k > 0).sum()) for k in ks) / total


def test_noise_ratio_reaches_workers(tmp_path):
    data = make(tmp_path / "toy", persons=20, seed=0)
    cfg = load_config(ROOT / "configs" / "smoke_toy.yaml",
                      [f"data.json={data.as_posix()}", f"data.image_root={(tmp_path / 'toy' / 'imgs').as_posix()}",
                       "masking.prob=0"])
    ds = PedesTrain(cfg, seed=0)
    assert _corrupted_fraction(ds, 0, 0.0) == 0.0
    assert _corrupted_fraction(ds, 1, 1.0) > 0.95
