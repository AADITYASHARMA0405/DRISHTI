import torch
import torch.nn.functional as F

from drishti.metrics import retrieval_metrics
from drishti.models.losses import robust_triplet_loss, sdm_loss


def test_perfect_retrieval():
    g_pids = torch.tensor([0, 1, 2, 3])
    q_pids = torch.tensor([0, 1, 2, 3])
    m = retrieval_metrics(torch.eye(4), q_pids, g_pids)
    assert m["R1"] == m["mAP"] == m["mINP"] == 100.0


def test_known_example():
    # one query (pid 0); gallery ranked [pid1, pid0, pid2, pid0]
    sim = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    m = retrieval_metrics(sim, torch.tensor([0]), torch.tensor([1, 0, 2, 0]))
    assert m["R1"] == 0.0 and m["R5"] == 100.0
    # AP = (1/2 + 2/4) / 2 = 0.5 ; INP = 2 relevant / last hit at rank 4 = 0.5
    assert abs(m["mAP"] - 50.0) < 1e-6
    assert abs(m["mINP"] - 50.0) < 1e-6


def _feats(n=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_sdm_lower_when_aligned():
    img = _feats()
    pid = torch.arange(8)
    aligned = sdm_loss(img, img, pid, torch.tensor(50.0))
    random_ = sdm_loss(img, _feats(seed=1), pid, torch.tensor(50.0))
    assert aligned > -1e-6 and aligned < random_   # KL is >= 0 up to float rounding


def test_robust_margin_shrinks_with_corruption():
    img, txt = _feats(), _feats(seed=1)
    pid = torch.arange(8)
    A = torch.full((8,), 4.0)
    clean = robust_triplet_loss(img, txt, pid, torch.zeros(8), A, margin=0.5)
    noisy = robust_triplet_loss(img, txt, pid, torch.full((8,), 3.0), A, margin=0.5)
    assert noisy < clean


def test_robust_ignores_same_person_negatives():
    img = _feats(4)
    pid = torch.tensor([0, 0, 0, 0])      # no negatives at all
    loss = robust_triplet_loss(img, img, pid, torch.zeros(4), torch.ones(4))
    assert loss == 0
