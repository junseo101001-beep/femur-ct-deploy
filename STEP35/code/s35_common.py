# -*- coding: utf-8 -*-
"""STEP35: only B0 contour/SDF pre-processing differs across M0/M1/M2."""
import sys, os, io, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "STEP34", "code"))
import s34_common as C34
C33 = C34.C33
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from phase6.phase6d_validation_drr import PIX, DET_W_MM, DET_H_MM, N_CONTOUR

VAL, TAGS = C33.VAL, C33.TAGS
MODELS = ["M0", "M1", "M2"]
CONDS = ["clean"] + [C34.cname(s, r) for s in (120, 80, 60, 50) for r in C34.RS]
OUT = os.path.join("STEP35", "results")

def background_stats(img):
    # Corners are detector background for this fixed DRR geometry.
    k = 40; a = np.concatenate((img[:k,:k].ravel(), img[:k,-k:].ravel(), img[-k:,:k].ravel(), img[-k:,-k:].ravel())).astype(float)
    med = float(np.median(a)); sig = float(1.4826 * np.median(np.abs(a - med)))
    return med, sig

def processed(img, model):
    x = img.astype(float)
    med, sig = background_stats(x)
    if model == "M0":
        return x, 0.02, med, sig
    if model == "M1":
        return x, max(0.02, med + 3.0 * sig), med, sig
    if model == "M2":
        return ndi.median_filter(x, size=3, mode="nearest"), 0.02, med, sig
    raise ValueError(model)

def contour_from_processed(x, threshold):
    """Exact contour_of tail, with only input threshold supplied by STEP35."""
    W, H = int(round(DET_W_MM / PIX)), int(round(DET_H_MM / PIX))
    m = x > threshold
    raw_components = int(ndi.label(m)[1])
    m = ndi.binary_closing(m, np.ones((3, 3)))
    m = ndi.binary_fill_holes(m)
    lab, nl = ndi.label(m)
    if nl:
        sz = ndi.sum(m, lab, range(1, nl + 1)); m = lab == (int(np.argmax(sz)) + 1)
    cs = measure.find_contours(m.astype(float), 0.5)
    if not cs: raise RuntimeError("no contour")
    c = max(cs, key=len)
    uv = np.stack([(c[:,1] - W/2.0)*PIX, (H/2.0 - c[:,0])*PIX], 1)
    p = np.vstack([uv, uv[:1]]); seg = np.sqrt(((p[1:] - p[:-1])**2).sum(1)); s = np.r_[0, np.cumsum(seg)]
    tt = np.linspace(0, s[-1], N_CONTOUR, endpoint=False)
    return np.stack([np.interp(tt,s,p[:,0]), np.interp(tt,s,p[:,1])], 1), m, raw_components

def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, io.open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
