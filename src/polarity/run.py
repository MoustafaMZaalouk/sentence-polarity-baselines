"""Run all models over several seeds; write results/results.json and plots."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .data import ensure_data, load_texts
from .metrics import accuracy, bootstrap_ci, expected_calibration_error
from .models import MODEL_NAMES, make_model

SEEDS = [0, 1, 2, 3, 4]


def split(texts, labels, seed):
    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    return tr, te


def run(seeds=SEEDS, subsample=None, out_dir="results", data_dir="data"):
    ensure_data(data_dir)
    texts, labels = load_texts(data_dir)
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)
    if subsample:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(texts), size=subsample, replace=False)
        texts, labels = texts[keep], labels[keep]
    per_model = {m: {"acc": [], "ece": []} for m in MODEL_NAMES}
    pooled = {m: ([], []) for m in MODEL_NAMES}
    errors = []
    t0 = time.time()
    for seed in seeds:
        tr, te = split(texts, labels, seed)
        for name in MODEL_NAMES:
            model = make_model(name, seed)
            model.fit(texts[tr], labels[tr])
            pred = model.predict(texts[te])
            per_model[name]["acc"].append(accuracy(labels[te], pred))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(texts[te])[:, 1]
                per_model[name]["ece"].append(expected_calibration_error(labels[te], proba))
            pooled[name][0].extend(labels[te].tolist())
            pooled[name][1].extend(pred.tolist())
            if name == "lr_word12_char" and seed == seeds[0]:
                wrong = np.where(pred != labels[te])[0]
                conf = np.abs(proba[wrong] - 0.5)
                for j in wrong[np.argsort(-conf)][:15]:
                    errors.append({"text": str(texts[te][j]), "label": int(labels[te][j]),
                                   "p_pos": round(float(proba[j]), 3)})
    summary = {}
    for name in MODEL_NAMES:
        a = np.array(per_model[name]["acc"])
        e = per_model[name]["ece"]
        lo, hi = bootstrap_ci(*pooled[name])
        summary[name] = {
            "accuracy_mean": round(float(a.mean()), 4),
            "accuracy_std_over_seeds": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "accuracy_pooled_bootstrap95": [round(lo, 4), round(hi, 4)],
            "ece_mean": round(float(np.mean(e)), 4) if e else None,
            "per_seed_accuracy": [round(float(x), 4) for x in a],
        }
    result = {
        "dataset": "Pang & Lee sentence polarity v1.0",
        "n_examples_used": int(len(texts)),
        "seeds": list(seeds),
        "protocol": "80/20 stratified random split per seed; hyperparameters fixed in advance, not tuned on the test split",
        "subsample_smoke_run": bool(subsample),
        "models": summary,
        "most_confident_errors_lr_word12_char_seed0": errors,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    plot(summary, out / "accuracy.png")
    return result


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summary)
    means = [summary[n]["accuracy_mean"] for n in names]
    errs = [summary[n]["accuracy_std_over_seeds"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(names, means, xerr=errs, color="#4c78a8")
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("test accuracy (mean, error bar = std over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None, help="smoke run on N sentences")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    r = run(SEEDS[: a.seeds], a.subsample, a.out)
    for n, s in r["models"].items():
        print(f"{n:24s} acc={s['accuracy_mean']:.4f} ece={s['ece_mean']}")


if __name__ == "__main__":
    main()
