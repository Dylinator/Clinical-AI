"""
model.py — fit / predict / persist ONLY (Rule 5).

Baselines live in baselines.py; metrics and threshold selection in evaluate.py.
(The earlier stub had a stray `from pyexpat import features` auto-import — that's
Python's XML parser, unrelated to this project — and a hand-written mega-list of
undefined feature names. Both are gone: the feature vector is built by
features.py, and the columns are fixed in config.FEATURES.)

The estimator is a scikit-learn Pipeline (preprocessing + classifier) wrapped in
CalibratedClassifierCV, fitted once and saved to artifacts/. Calibration matters
because the dashboard shows a probability as a percentage — "72%" has to mean 72%.
Swap RandomForest for XGBoost here later without touching anything else.
"""

from __future__ import annotations
import os
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import config


def build_estimator():
    base = Pipeline([
        # Scaling is a no-op for trees but keeps the pipeline swap-ready for
        # scale-sensitive models; the point is that preprocessing travels WITH
        # the estimator so training and serving preprocess identically (Rule 1).
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        )),
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def fit(X, y):
    est = build_estimator()
    est.fit(X, y)
    return est


def predict_risk(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


def save(est, path: str = config.PIPELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(est, path)


def load(path: str = config.PIPELINE_PATH):
    return joblib.load(path)


def global_importances(est, feature_names=config.FEATURES):
    """Average RandomForest importances across calibration folds (best-effort).
    A lightweight stand-in for SHAP until you wire real per-step attributions."""
    try:
        imps = [cc.estimator.named_steps["rf"].feature_importances_
                for cc in est.calibrated_classifiers_]
        imp = np.mean(imps, axis=0)
        order = np.argsort(imp)[::-1]
        return [(feature_names[i], float(imp[i])) for i in order]
    except Exception:
        return [(f, 0.0) for f in feature_names]