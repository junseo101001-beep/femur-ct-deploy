# -*- coding: utf-8 -*-
"""STEP 33 공통 : 영상 품질 오염 조건 (사전 등록 §2 고정값), 오염 DRR·contour·QC, B0 입력 블록, drr_g 경로 우회."""
import sys, os, io, json
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
for p in (_H, ROOT, os.path.join(ROOT, "STEP26", "code"), os.path.join(ROOT, "STEP23", "code"), os.path.join(ROOT, "STEP21", "code"),
          os.path.join(ROOT, "STEP8", "scripts"), os.path.join(ROOT, "STEP9", "code"), os.path.join(ROOT, "STEP10", "code"), os.path.join(ROOT, "STEP12", "code")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import numpy as np

VAL = ["Pat019", "Pat031", "Pat042", "Pat060", "Pat080", "Pat095"]
NOM = [0.0, 45.0, 90.0]
TAGS = ["000.0deg", "045.0deg", "090.0deg"]
MU_FG, S_FG, GMAX = 0.7808568000325599, 0.3187897897504893, 2.484509229660034
SIG = {1: 0.02 * S_FG, 2: 0.05 * S_FG, 3: 0.10 * S_FG}
N0_POISSON = float(np.exp(MU_FG) / SIG[2] ** 2)
CONDS = {"clean": {"factor": "clean"},
         "gn_low_r0": {"factor": "noise", "level": 1, "r": 0}, "gn_low_r1": {"factor": "noise", "level": 1, "r": 1},
         "gn_med_r0": {"factor": "noise", "level": 2, "r": 0}, "gn_med_r1": {"factor": "noise", "level": 2, "r": 1},
         "gn_high_r0": {"factor": "noise", "level": 3, "r": 0}, "gn_high_r1": {"factor": "noise", "level": 3, "r": 1},
         "poisson_med": {"factor": "poisson"},
         "blur_low": {"factor": "blur", "sigma_px": 1.0}, "blur_med": {"factor": "blur", "sigma_px": 2.0},
         "con_low": {"factor": "contrast", "c": 0.5}, "con_mid": {"factor": "contrast", "c": 0.75}, "con_high": {"factor": "contrast", "c": 1.5},
         "composite": {"factor": "composite", "c": 0.75, "sigma_px": 1.0, "level": 1}}
OUT = os.path.join("STEP33", "results")
DRR = os.path.join(OUT, "drr")


def gauss_noise(shape, si, vi, level, r):
    return np.random.default_rng([20261501, si, vi, level, r]).standard_normal(shape)


def degrade(img, cond, si, vi):
    import scipy.ndimage as ndi
    s = CONDS[cond]; f = s["factor"]; x = img.astype(np.float64)
    if f == "clean":
        return img
    if f == "noise":
        return (x + SIG[s["level"]] * gauss_noise(x.shape, si, vi, s["level"], s["r"])).astype(np.float32)
    if f == "poisson":
        lam = N0_POISSON * np.exp(-x)
        N = np.random.default_rng([20261502, si, vi]).poisson(lam)
        return (-np.log(np.maximum(N, 1) / N0_POISSON)).astype(np.float32)
    if f == "blur":
        return ndi.gaussian_filter(x, s["sigma_px"], mode="nearest", truncate=4.0).astype(np.float32)
    if f == "contrast":
        return np.clip(s["c"] * x, 0, GMAX).astype(np.float32)
    if f == "composite":
        y = np.clip(s["c"] * x, 0, GMAX)
        y = ndi.gaussian_filter(y, s["sigma_px"], mode="nearest", truncate=4.0)
        return (y + SIG[1] * np.random.default_rng([20261501, si, vi, 9, 0]).standard_normal(y.shape)).astype(np.float32)
    raise ValueError(cond)


_D = {}


def D17():
    if not _D:
        import s21_common as C21
        from phase6.phase6_ssm_xray import project
        from phase5_correspond import read_palp
        from phase6.phase6h_extend import _imaging_of
        _D.update(C21.extract(os.path.join("STEP17", "code", "s17_03_descriptors.py"), ["ANG", "frame", "norm", "knee_2d"],
                              {"np": np, "os": os, "read_palp": read_palp, "_imaging_of": _imaging_of, "project": project}))
    return _D


def blocks_and_qc(pid, views):
    """views : TAGS 순서 view dict. STEP26 s26_02 와 같은 SDF·LMK 블록과 입력 검사."""
    import s23_common as CM
    D = D17(); kn = D["knee_2d"](pid)
    s_v, l_v, chk = [], [], []
    for t, a in zip(TAGS, D["ANG"]):
        v = views[t]
        C = np.array(v["contour_uv_mm"], float)
        L3 = [np.array(v["landmarks_2d_mm"]["femurHead"]), np.array(v["landmarks_2d_mm"]["greatTroch"]), kn[a]]
        c, R, Lz = D["frame"](C, L3[0])
        Q = D["norm"](C, c, R, Lz)
        s_v.append(CM.sdf_grid(Q))
        ln = [D["norm"](x, c, R, Lz) for x in L3]
        l_v.append(np.concatenate(ln))
        chk.append({"frac_contour_outside_sdf_grid": float(((Q[:, 0] < CM.U_GRID[0]) | (Q[:, 0] > CM.U_GRID[-1]) | (Q[:, 1] < CM.V_GRID[0]) | (Q[:, 1] > CM.V_GRID[-1])).mean()),
                    "landmarks_outside_detector": [bool(abs(x[0]) > 180 or abs(x[1]) > 240) for x in L3], "head_above": bool(ln[0][1] > 0), "knee_below": bool(ln[2][1] < 0),
                    "projected_length_mm": float(Lz)})
    ok = all(c["frac_contour_outside_sdf_grid"] == 0.0 and c["head_above"] and c["knee_below"] and not any(c["landmarks_outside_detector"]) for c in chk)
    return np.array(s_v), np.concatenate(l_v), chk, bool(ok)


class redirect_drr_g(object):
    """os.path.join 첫 인자가 정확히 'drr_g' 일 때만 target 으로 (메모리에서만)."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        import ntpath
        self._orig = os.path.join
        orig, tgt = self._orig, self.target

        def _join(a, *p):
            return orig(tgt if a == "drr_g" else a, *p)
        os.path.join = _join
        if os.path is ntpath:
            ntpath.join = _join
        return self

    def __exit__(self, *a):
        import ntpath
        os.path.join = self._orig
        if os.path is ntpath:
            ntpath.join = self._orig


def jdump(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(o, io.open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
