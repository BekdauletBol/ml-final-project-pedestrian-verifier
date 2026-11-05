"""Model 1 – classical machine learning: hand-made features + SVM.

Before deep learning, pedestrians were usually recognized with HOG features
(histograms of oriented gradients) and a linear SVM (Dalal & Triggs, 2005).
I use this as my classical model and compare it with my CNN and MobileNetV2.

The gray crops are made with OpenCV, the HOG features with scikit-image and the
classifiers come from scikit-learn.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

HOG_SIZE = 64  # the crop is resized to 64x64 gray before HOG
# cell 8x8 px, block 2x2 cells (moves by one cell), 9 orientation bins, L2-Hys block norm
# like Dalal & Triggs -> 7 x 7 blocks x 36 values = 1,764 features
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


def to_gray(x_uint8, size: int) -> np.ndarray:
    """(N, 3, H, W) uint8 tensor or array -> (N, size, size) uint8 gray images."""
    x = x_uint8.numpy() if isinstance(x_uint8, torch.Tensor) else np.asarray(x_uint8)
    out = np.empty((len(x), size, size), dtype=np.uint8)
    for i, img in enumerate(x):
        rgb = np.ascontiguousarray(img.transpose(1, 2, 0))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return out


def pixel_features(x_uint8, size: int = 32) -> np.ndarray:
    """Baseline features: the gray pixels themselves (size*size values in [0, 1])."""
    return to_gray(x_uint8, size).reshape(len(x_uint8), -1).astype(np.float32) / 255.0


def _hog_chunk(gray: np.ndarray) -> np.ndarray:
    return np.stack([hog(g, feature_vector=True, **HOG_PARAMS) for g in gray]).astype(np.float32)


def hog_features(x_uint8, n_jobs: int = -1) -> np.ndarray:
    """HOG descriptor of every crop: (N, 1764) float32 (computed in parallel chunks)."""
    gray = to_gray(x_uint8, HOG_SIZE)
    chunks = np.array_split(gray, max(1, len(gray) // 1000))
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_hog_chunk)(c) for c in chunks))


def make_pipeline(kind: str, C: float = 1.0, k: int = 500, seed: int = 42) -> Pipeline:
    """The classical pipelines I compare. All start with standardization (z-score)."""
    steps = [("scale", StandardScaler())]
    if kind == "linear_svm":
        steps.append(("clf", LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed)))
    elif kind == "anova_linear_svm":  # feature selection: keep the k features with the highest ANOVA F
        steps += [("select", SelectKBest(f_classif, k=k)),
                  ("clf", LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed))]
    elif kind == "pca_linear_svm":  # dimensionality reduction: keep 95 % of the variance
        steps += [("pca", PCA(n_components=0.95, random_state=seed)),
                  ("clf", LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed))]
    elif kind == "pca_rbf_svm":
        steps += [("pca", PCA(n_components=0.95, random_state=seed)),
                  ("clf", SVC(C=C, kernel="rbf", gamma="scale", random_state=seed))]
    elif kind == "pca_logreg":
        steps += [("pca", PCA(n_components=0.95, random_state=seed)),
                  ("clf", LogisticRegression(C=C, max_iter=3000))]
    elif kind == "random_forest":
        steps = [("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed))]
    else:
        raise ValueError(kind)
    return Pipeline(steps)
