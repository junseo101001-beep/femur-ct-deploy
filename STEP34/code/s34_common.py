# -*- coding: utf-8 -*-
"""STEP 34 공통 : SNR 격자 (σ = μ_fg/SNR, STEP33 정의), seed, 조건 이름. STEP33 공통 (QC·B0 블록·경로 우회) 를 읽기 전용 import."""
import sys, os, io, json
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
sys.path.insert(0, os.path.join(ROOT, "STEP33", "code"))
sys.path.insert(0, _H)
import s33_common as C33                       # 경로 설정·chdir 포함
import numpy as np

VAL, TAGS = C33.VAL, C33.TAGS
ST33 = json.load(io.open(os.path.join("STEP33", "results", "intensity_stats.json"), encoding="utf-8"))
MU_FG = ST33["foreground_pooled"]["mean"]
assert MU_FG == C33.MU_FG == 0.7808568000325599
SNRS = [120, 100, 80, 70, 60, 55, 50, 45, 40]
SIGMA = {s: MU_FG / s for s in SNRS}
RS = [0, 1]
OUT = os.path.join("STEP34", "results")
DRR = os.path.join(OUT, "drr")


def cname(snr, r):
    return "snr%03d_r%d" % (snr, r)


CONDS = ["clean"] + [cname(s, r) for s in SNRS for r in RS]


def parse(c):
    if c == "clean":
        return None, None
    return int(c[3:6]), int(c[-1])


def noisy(img, snr, si, vi, r):
    return (img.astype(np.float64) + SIGMA[snr] * np.random.default_rng([20261601, snr, si, vi, r]).standard_normal(img.shape)).astype(np.float32)


def jdump(o, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(o, io.open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
