"""
Numpy-only Random Forest inference.

scikit-learn + SciPy add ~200 MB, which doesn't fit a serverless bundle
(Vercel's Python limit is 500 MB). The trained forest is exported once to a
flat .npz (`export_forest`) and evaluated here with plain numpy — identical
predictions, no sklearn at runtime.
"""
from __future__ import annotations

import numpy as np


class NumpyForestRegressor:
    """Mean of decision-tree regressors stored as flat node arrays."""

    def __init__(self, left, right, feature, threshold, value, roots, n_features):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.roots = roots
        self.n_features_in_ = int(n_features)

    @classmethod
    def load(cls, path: str) -> "NumpyForestRegressor":
        d = np.load(path)
        return cls(d["left"], d["right"], d["feature"], d["threshold"], d["value"],
                   d["roots"], d["n_features"])

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        out = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            total = 0.0
            for root in self.roots:
                node = int(root)
                while self.left[node] != -1:
                    # sklearn: go left when x <= threshold (thresholds are float64,
                    # inputs are cast to float32 first)
                    if np.float32(row[self.feature[node]]) <= self.threshold[node]:
                        node = int(self.left[node])
                    else:
                        node = int(self.right[node])
                total += self.value[node]
            out[i] = total / len(self.roots)
        return out


def export_forest(model, path: str) -> None:
    """Flatten a fitted sklearn RandomForestRegressor into `path` (.npz)."""
    lefts, rights, feats, thrs, vals, roots = [], [], [], [], [], []
    offset = 0
    for est in model.estimators_:
        t = est.tree_
        n = t.node_count
        left = t.children_left.astype(np.int64)
        right = t.children_right.astype(np.int64)
        lefts.append(np.where(left == -1, -1, left + offset))
        rights.append(np.where(right == -1, -1, right + offset))
        feats.append(t.feature.astype(np.int64))
        thrs.append(t.threshold.astype(np.float64))
        vals.append(t.value[:, 0, 0].astype(np.float64))
        roots.append(offset)
        offset += n
    np.savez_compressed(
        path,
        left=np.concatenate(lefts), right=np.concatenate(rights),
        feature=np.concatenate(feats), threshold=np.concatenate(thrs),
        value=np.concatenate(vals), roots=np.array(roots, dtype=np.int64),
        n_features=np.array(model.n_features_in_),
    )


def load_rf_model(models_dir):
    """The ASTRA.AI forest: the sklearn pickle when scikit-learn is installed,
    else the numpy export (serverless). None if neither is available."""
    import os
    joblib_path = os.path.join(str(models_dir), "astra_rf.joblib")
    npz_path = os.path.join(str(models_dir), "astra_rf_trees.npz")
    try:
        import sklearn  # noqa: F401
        import joblib
        if os.path.exists(joblib_path):
            return joblib.load(joblib_path)
    except ImportError:
        pass
    if os.path.exists(npz_path):
        return NumpyForestRegressor.load(npz_path)
    return None
