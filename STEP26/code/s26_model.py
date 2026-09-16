# -*- coding: utf-8 -*-
"""STEP 26 : 잠금 모델 파일에서 B0 추론 (STEP23 Joint.proj / lift_3d 와 같은 식). validation·training 공통."""
import numpy as np

BLOCKS_2D = ["SDF_000.0deg", "SDF_045.0deg", "SDF_090.0deg", "LMK"]


def load_model(path):
    Z = np.load(path, allow_pickle=False)
    M = {k: Z[k] for k in Z.files}
    M["K"] = int(M["K"]); M["gamma"] = float(M["gamma"])
    return M


def predict(M, sdf, lmk):
    """sdf : (n, 3, 112, 40), lmk : (n, 18) → 정규화 해부 frame 형상 (n, 4911, 3)."""
    n = len(sdf)
    raw = {"SDF_000.0deg": sdf[:, 0].reshape(n, -1), "SDF_045.0deg": sdf[:, 1].reshape(n, -1), "SDF_090.0deg": sdf[:, 2].reshape(n, -1), "LMK": lmk}
    K, g = M["K"], M["gamma"]
    X2 = np.hstack([(raw[b] - M["mean_" + b]) / M["sd_" + b] / M["sqrtd_" + b] for b in BLOCKS_2D])
    Vt = M["Vt"]
    W = np.hstack([Vt[:K, int(M["slice_" + b][0]):int(M["slice_" + b][1])] for b in BLOCKS_2D])
    A = W @ W.T + g * np.diag(1.0 / np.maximum(M["var"][:K], 1e-12))
    z = np.linalg.solve(A, W @ np.atleast_2d(X2).T).T
    s3 = slice(int(M["slice_3D"][0]), int(M["slice_3D"][1]))
    X3 = M["mean_3D"] + (z @ Vt[:K, s3]) * M["sd_3D"] * M["sqrtd_3D"]
    return X3.reshape(n, -1, 3), z, float(np.linalg.cond(A))
