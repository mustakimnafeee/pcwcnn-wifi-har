"""
baselines.py  -  the paper's reference models (section 4.3), used for Fig. 7, Table 1, Fig. 8.

  SVM  : RBF kernel, C = 11
  RF   : random forest, 200 trees, maximum depth 20
  KNN  : k = 3 nearest neighbours, Euclidean distance
  CNN  : a small 1D-CNN that "takes a 384-element vector as input"

Inputs are the DWT (wavelet) features of the preprocessed principal components:

  * SVM / RF / KNN get the approximation coefficients of all 3 DWT levels of every PC
    (192 + 96 + 48 = 336 numbers per PC), standardised.
  * The 1D-CNN gets the level-1 approximation of each PC side by side. With the paper's
    two PCs that is 192 + 192 = 384, exactly the "384-element vector" the paper describes
    (other PC counts are zero-padded or cut to 384).

These are built the same way as in the full project, where they reproduced the paper's
Table 1 to within about 1 point.
"""
import numpy as np
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model import conv, train_and_test

CLASSICAL = ["SVM", "RF", "KNN"]
ALL_MODELS = CLASSICAL + ["CNN"]


# ---------------------------------------------------------------- DWT features
def haar_levels(x, levels=3):
    """Haar DWT approximations of x (..., L): [level 1 (L/2), level 2 (L/4), level 3 (L/8)]."""
    out = []
    for _ in range(levels):
        x = (x[..., 0::2] + x[..., 1::2]) / np.sqrt(2)
        out.append(x)
    return out


def dwt_features(X):
    """X (n, n_pcs, 384) -> (n, n_pcs * 336) features for SVM / RF / KNN."""
    return np.concatenate(
        [np.concatenate(haar_levels(X[:, c]), axis=1) for c in range(X.shape[1])], axis=1)


def cnn_vector(X):
    """X (n, n_pcs, 384) -> (n, 1, 384): level-1 approximation of each PC, side by side."""
    F = np.concatenate([haar_levels(X[:, c], 1)[0] for c in range(X.shape[1])], axis=1)
    if F.shape[1] >= 384:
        F = F[:, :384]
    else:
        F = np.pad(F, ((0, 0), (0, 384 - F.shape[1])))
    return F[:, None, :].astype(np.float32)


# ---------------------------------------------------------------- the models
def classical_model(name, seed=0):
    """SVM, RF or KNN with the paper's settings (inputs standardised first)."""
    if name == "SVM":
        clf = SVC(kernel="rbf", C=11, gamma="scale", random_state=seed)
    elif name == "RF":
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed,
                                     n_jobs=-1)
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), clf)


class Reference1DCNN(nn.Module):
    """The paper's reference CNN: 4 conv layers + 2 fully-connected layers."""

    def __init__(self, n_classes=16):
        super().__init__()
        self.features = nn.Sequential(
            conv(1, 32), nn.MaxPool1d(2),       # 384 -> 192
            conv(32, 64), nn.MaxPool1d(2),      # 192 -> 96
            conv(64, 128), nn.MaxPool1d(2),     # 96  -> 48
            conv(128, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 48, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_reference_models(Xtr, ytr, Xte, yte, seed=0, models=ALL_MODELS, cnn_epochs=80):
    """Train each reference model on (Xtr, ytr), return {model name: test predictions}."""
    preds = {}
    wanted = [m for m in models if m in CLASSICAL]
    if wanted:
        Ftr, Fte = dwt_features(Xtr), dwt_features(Xte)
        for name in wanted:
            preds[name] = classical_model(name, seed).fit(Ftr, ytr).predict(Fte)
    if "CNN" in models:
        pred, _, _ = train_and_test(cnn_vector(Xtr), ytr, cnn_vector(Xte), yte,
                                    epochs=cnn_epochs, use_augment=False, seed=seed,
                                    make_model=Reference1DCNN, verbose=False)
        preds["CNN"] = pred
    return preds
