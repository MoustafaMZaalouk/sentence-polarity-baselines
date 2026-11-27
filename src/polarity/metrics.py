import numpy as np


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def expected_calibration_error(y, prob_pos, bins=10):
    """Binary ECE on the predicted-class confidence."""
    y = np.asarray(y)
    prob_pos = np.asarray(prob_pos, dtype=float)
    pred = (prob_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, prob_pos, 1 - prob_pos)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    accs = (y[idx] == p[idx]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))
