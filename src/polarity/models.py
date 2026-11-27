"""Model zoo: all linear models on sparse n-gram features, CPU only."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def make_model(name, seed=0):
    if name == "majority":
        from sklearn.dummy import DummyClassifier
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "nb_unigram":
        return Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 1), min_df=1, use_idf=False, norm=None)),
            ("clf", MultinomialNB(alpha=1.0)),
        ])
    if name == "lr_word12":
        return Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=seed)),
        ])
    if name == "lr_word12_char":
        feats = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
        ])
        return Pipeline([
            ("tf", feats),
            ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=seed)),
        ])
    if name == "svm_word12_calibrated":
        return Pipeline([
            ("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, random_state=seed), cv=3)),
        ])
    raise ValueError(name)


MODEL_NAMES = ["majority", "nb_unigram", "lr_word12", "lr_word12_char", "svm_word12_calibrated"]
