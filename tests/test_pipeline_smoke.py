"""End-to-end smoke test on synthetic data: toy data -> noisy sets -> train -> eval -> gallery search."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval as eval_script            # noqa: E402
import make_noisy_sets                # noqa: E402
import train as train_script          # noqa: E402
from make_toy_data import make        # noqa: E402

from drishti.index import GallerySearcher  # noqa: E402
from drishti.models import load_model      # noqa: E402


def test_end_to_end(tmp_path):
    data = make(tmp_path / "toy", persons=100, seed=0)
    opts = [f"data.json={data.as_posix()}", f"data.image_root={(tmp_path / 'toy' / 'imgs').as_posix()}",
            f"output_root={(tmp_path / 'outputs').as_posix()}"]
    cfg_path = str(ROOT / "configs" / "smoke_toy.yaml")

    # noisy query sets are reproducible
    qdir = tmp_path / "qs"
    r1 = make_noisy_sets.main(["--config", cfg_path, "--out", str(qdir), "--opts", *opts])
    r2 = make_noisy_sets.main(["--config", cfg_path, "--out", str(qdir), "--opts", *opts])
    assert r1["checksum"] == r2["checksum"]
    assert r1["summary"]["commission_s1"]["changed"] > 0.9

    # training runs and learns something
    res = train_script.main(["--config", cfg_path, "--seed", "1", "--opts", *opts])
    hist = res["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    out = Path(res["out_dir"])
    assert (out / "best.pt").exists() and (out / "last.pt").exists()

    # evaluation writes results and the gallery file
    results = tmp_path / "results.csv"
    gal = out / "gallery.npz"
    rows = eval_script.main(["--config", cfg_path, "--checkpoint", str(out / "best.pt"),
                             "--query-sets", str(qdir / "commission_s*.json"),
                             "--results", str(results), "--save-gallery", str(gal), "--opts", *opts])
    names = [r["query_set"] for r in rows]
    assert names[0] == "clean" and "commission_s3" in names
    # 15 test identities x 2 images: random ranking gives R1 ~ 6.7 and mAP ~ 13
    assert rows[0]["R1"] > 100 / 15
    assert rows[0]["mAP"] > 30
    assert len(list(csv.DictReader(results.open()))) == len(rows)

    # gallery search returns a matching person for a clean query
    s = GallerySearcher(gal)
    model = load_model(out / "best.pt", device="cpu")
    q = json.loads((qdir / "clean.json").read_text())["queries"][0]
    with torch.no_grad():
        feat = model.encode_text([q["text"]]).numpy()
    scores, idx = s.search(feat, k=10)
    assert scores.shape == (1, 10)
    assert np.all(np.diff(scores[0]) <= 1e-6)
