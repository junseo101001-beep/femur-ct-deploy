# -*- coding: utf-8 -*-
"""STEP 37 공통 : 잠금 B0 파일 기반 F1 추론 (STEP29 F1 정의, s26_model.predict 와 같은 식) 과 B0 missing 경로 (STEP27/29 : GT·KC = training 평균)."""
import os, sys, io, json
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
for p in (_H, os.path.join(ROOT, "STEP26", "code")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

SDFB = ["SDF_000.0deg", "SDF_045.0deg", "SDF_090.0deg"]
FH = np.array([0, 1, 6, 7, 12, 13])                 # s29_common.FH (femurHead u,v × 3 view)
GKC = np.array([2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17])
VAL = ["Pat019", "Pat031", "Pat042", "Pat060", "Pat080", "Pat095"]
OUT = os.path.join("STEP37", "results")


def _norm(M, b, x):
    return (x - M["mean_" + b]) / M["sd_" + b] / M["sqrtd_" + b]


def _sl(M, b):
    return slice(int(M["slice_" + b][0]), int(M["slice_" + b][1]))


def predict_f1(M, sdf, lmk, K, g):
    """sdf (n,3,112,40), lmk (n,18) ; GT·KC 값은 사용하지 않음 (NaN 이어도 됨)."""
    n = len(sdf)
    X2 = np.hstack([_norm(M, b, sdf[:, v].reshape(n, -1)) for v, b in enumerate(SDFB)] + [_norm(M, "LMK", lmk)[:, FH]])
    Vt = M["Vt"]
    W = np.hstack([Vt[:K, _sl(M, b)] for b in SDFB] + [Vt[:K, _sl(M, "LMK")][:, FH]])
    A = W @ W.T + g * np.diag(1.0 / np.maximum(M["var"][:K], 1e-12))
    z = np.linalg.solve(A, W @ np.atleast_2d(X2).T).T
    X3 = M["mean_3D"] + (z @ Vt[:K, _sl(M, "3D")]) * M["sd_3D"] * M["sqrtd_3D"]
    return X3.reshape(n, -1, 3), z, float(np.linalg.cond(A))


def b0_missing_lmk(M, lmk):
    """STEP27 N7 / STEP28 T6 : GT·KC 12 열을 training 평균으로 (정규화 후 0)."""
    x = np.array(lmk, float).copy()
    x[..., GKC] = M["mean_LMK"][GKC]
    return x


def jdump(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(o, io.open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
