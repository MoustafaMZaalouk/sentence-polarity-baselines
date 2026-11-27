"""Download, verify and load the Pang & Lee sentence polarity dataset v1.0."""
import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
SHA256 = "a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9"
DATA_DIR = Path("data")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    pos = data_dir / "rt-polaritydata" / "rt-polarity.pos"
    neg = data_dir / "rt-polaritydata" / "rt-polarity.neg"
    if pos.exists() and neg.exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "rt-polaritydata.tar.gz"
    urllib.request.urlretrieve(URL, tgz)
    got = sha256_of(tgz)
    if got != SHA256:
        tgz.unlink()
        raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {got}")
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = Path(m.name)
            if m.name.startswith(("/", "..")) or ".." in name.parts or not m.isfile():
                raise RuntimeError(f"unsafe archive member: {m.name}")
        tf.extractall(data_dir)
    return data_dir


def load_texts(data_dir=DATA_DIR):
    """Return (texts, labels); label 1 = positive, 0 = negative."""
    data_dir = Path(data_dir)
    texts, labels = [], []
    for fname, label in (("rt-polarity.pos", 1), ("rt-polarity.neg", 0)):
        with open(data_dir / "rt-polaritydata" / fname, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(label)
    return texts, labels
