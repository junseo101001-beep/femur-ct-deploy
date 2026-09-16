# -*- coding: utf-8 -*-
"""STEP 21 공통 : 접근 감사, STEP17 함수 소스 추출 재사용, D2 계산, 지표."""
import sys, io, os, ast, json
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
for p in (ROOT, os.path.join(ROOT, "STEP18", "code")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import numpy as np

S21 = "STEP21"
HELD = ("Pat019", "Pat031", "Pat042", "Pat060", "Pat080", "Pat095", "Pat001")
FORBID = HELD + ("drr_g", "drr_val", "ground_truth", "phase6k_ssm_norm", "STEP12", "/validation", "\\validation")
CODE_OK = ("phase6d_validation_drr",)
ACCESS = []
TAGS = ["000.0deg", "045.0deg", "090.0deg"]


def install_audit():
    def hook(ev, a):
        if ev in ("open", "os.listdir", "os.scandir") and a and isinstance(a[0], (str, bytes, os.PathLike)):
            ACCESS.append(os.fsdecode(a[0]) if not isinstance(a[0], str) else a[0])
    sys.addaudithook(hook)


def audit_report():
    paths = sorted(set(ACCESS))
    return {"n_events": len(ACCESS), "forbidden_hits": [p for p in paths if any(t in p for t in FORBID) and not any(c in p for c in CODE_OK)],
            "project_data_paths": [p for p in paths if "femur-ct" in os.path.abspath(p) and "__pycache__" not in p and not p.endswith((".py", ".pyc"))]}


def extract(path, names, extra_ns):
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in keep}
    assert found == set(names), (path, set(names) - found)
    ns = dict(extra_ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def d2_ns():
    return extract(os.path.join("STEP17", "code", "s17_03_descriptors.py"), ["NB", "LM", "frame", "norm", "band_lr"], {"np": np})


def ridge_ns():
    return extract(os.path.join("STEP17", "code", "s17_04_ridge_loo.py"), ["LAMS", "fit", "predict"], {"np": np})


def d2_from(C_views, L_views, D):
    """C_views : [240×2 mm] × 3, L_views : [(3×2) mm, 순서 femurHead, greatTroch, kneeCenter] × 3 → D2 438, 검사."""
    d1, lm, chk = [], [], []
    for C, L3 in zip(C_views, L_views):
        c, R, Lz = D["frame"](C, L3[0])
        Q = D["norm"](C, c, R, Lz)
        lr, _, ne = D["band_lr"](Q)
        d1.append(lr)
        ln = [D["norm"](L3[k], c, R, Lz) for k in range(3)]
        lm.append(np.concatenate(ln))
        chk.append({"empty_bands": int(ne), "head_above": bool(ln[0][1] > 0), "knee_below": bool(ln[2][1] < 0), "length_mm": float(Lz)})
    return np.concatenate([np.concatenate(d1), np.concatenate(lm)]), chk


def coef_rmse(P, Y):
    return float(np.mean(np.sqrt(((P - Y) ** 2).mean(1))))


def pc_table(P, Y, mp_pc, loo_mp_pc=None):
    out = []
    for k in range(Y.shape[1]):
        e = P[:, k] - Y[:, k]
        rm = float(np.sqrt((e ** 2).mean())); sd = float(P[:, k].std())
        row = {"pc": k + 1, "rmse": rm, "ratio_to_mean_predictor": rm / mp_pc[k], "pred_sd": sd, "target_sd": float(Y[:, k].std()),
               "corr": float(np.corrcoef(P[:, k], Y[:, k])[0, 1]) if sd > 1e-12 else None,
               "r2": float(1 - (e ** 2).sum() / ((Y[:, k] - Y[:, k].mean()) ** 2).sum()), "mean_bias": float(e.mean())}
        if loo_mp_pc is not None:
            row["ratio_to_STEP19_loo_mean_predictor"] = rm / loo_mp_pc[k]
        out.append(row)
    return out


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, io.open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
