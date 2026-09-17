# -*- coding: utf-8 -*-
"""
Read-only visualization adapter for the locked reconstruction pipeline.

연구 코드를 수정하거나 복제하지 않는다. main.run_pipeline 과 **같은 연구 함수를 같은 순서로** 호출하고,
그 과정에서 이미 계산되는 중간 결과를 버리지 않고 모아서 돌려준다.

  01 input       DRR line-integral (drr_g/<pid>/drr_<tag>.npy)                      — 실제 배열
  02 threshold   s35_common.processed(img, "M1")  → T, bg median, 1.4826·MAD           — 실제 값
  03 contour     s35_common.contour_from_processed → mask(bool), contour_uv_mm(240×2)  — 실제 배열
  04 sdf         s33_common.blocks_and_qc          → SDF (3×112×40, 정규화 해부 frame)  — 실제 배열
  05 landmarks   case.json landmarks_2d_mm (femurHead, greatTroch) + STEP17 knee_2d    — 3D landmark 투영값
                 STEP17 frame/norm (blocks_and_qc 내부와 같은 함수) → 정규화 좌표 18개
  06 ssm         s26_model.predict (K=15, γ=0.001) / s37_common.predict_f1 (K=20, γ=0.01)
                 → latent z, var[:K], mean_3D, 추정 형상 X                              — 실제 값
  07 final       pose_apply(X, E0 pose) → display 변환 → STL                           — main 과 동일

추가로 계산하는 것은 표시용 좌표 변환(mm → pixel)과, blocks_and_qc 가 내부에서만 쓰는
정규화 윤곽 Q / 정규화 landmark 를 **같은 STEP17 함수**로 다시 얻는 것뿐이다.
그 정규화 landmark 는 blocks_and_qc 가 반환한 lmk 와 np.array_equal 로 매번 검사한다.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from typing import Any

import numpy as np

VIEW_TAGS = ["000.0deg", "045.0deg", "090.0deg"]
LM_NAMES = ["femurHead", "greatTroch", "kneeCenter"]
# main.run_pipeline 과 같은 GT·KC 열 (view 마다 [head u,v | troch u,v | knee u,v])
GKC = np.array([2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17])


def trace_pipeline(E: dict[str, Any], femur_root: str, pid: str, images: dict[str, np.ndarray], condition: str) -> dict[str, Any]:
    M1C, MD, C33, M = E["M1C"], E["MD"], E["C33"], E["model"]
    from phase6.phase6d_validation_drr import PIX, DET_W_MM, DET_H_MM
    W, H = int(round(DET_W_MM / PIX)), int(round(DET_H_MM / PIX))
    t0 = time.time()

    case = json.load(io.open(os.path.join(femur_root, "drr_g", pid, "case.json"), encoding="utf-8"))
    by = {v["tag"]: v for v in case["reconstruction_input"]["views"]}

    # ---- 02 threshold / 03 contour : main.run_pipeline 과 동일 호출
    views, per_view = {}, []
    for tag in VIEW_TAGS:
        x, thr, med, sig = M1C.processed(images[tag], "M1")
        cont, mask, ncomp = M1C.contour_from_processed(x, thr)
        views[tag] = {"view_angle_deg": by[tag]["view_angle_deg"], "tag": tag,
                      "landmarks_2d_mm": by[tag]["landmarks_2d_mm"],
                      "contour_uv_mm": np.round(cont, 4).tolist()}
        per_view.append({"tag": tag, "angle_deg": float(by[tag]["view_angle_deg"]),
                         "threshold": float(thr), "bg_median": float(med), "bg_mad_scale": float(sig),
                         "fixed_threshold": 0.02, "adaptive_used": bool(thr > 0.02 + 1e-12),
                         "raw_min": float(images[tag].min()), "raw_max": float(images[tag].max()),
                         "mask": mask, "mask_px": int(mask.sum()), "raw_components": int(ncomp),
                         "contour_uv_mm": np.round(cont, 4)})
    t_pre = time.time() - t0

    # ---- 04 SDF + 05 LMK : 추론에 실제로 들어가는 블록
    sdf, lmk, checks, qc_ok = C33.blocks_and_qc(pid, views)
    import s23_common as CM                              # SDF 격자 정의 (blocks_and_qc 와 같은 모듈)

    # 정규화 윤곽/landmark 를 blocks_and_qc 와 같은 STEP17 함수로 (표시용)
    D = C33.D17(); kn = D["knee_2d"](pid)
    lmk_check = []
    for pv, tag, a in zip(per_view, VIEW_TAGS, D["ANG"]):
        C = np.array(views[tag]["contour_uv_mm"], float)
        L3 = [np.array(views[tag]["landmarks_2d_mm"]["femurHead"]), np.array(views[tag]["landmarks_2d_mm"]["greatTroch"]), kn[a]]
        c, R, Lz = D["frame"](C, L3[0])
        Q = D["norm"](C, c, R, Lz)
        ln = [D["norm"](p, c, R, Lz) for p in L3]
        lmk_check.append(np.concatenate(ln))
        pv["contour_norm"] = Q
        pv["landmarks_mm"] = {n: np.asarray(p, float) for n, p in zip(LM_NAMES, L3)}
        pv["landmarks_px"] = {n: np.array([p[0] / PIX + W / 2.0, H / 2.0 - p[1] / PIX]) for n, p in zip(LM_NAMES, L3)}
        pv["landmarks_norm"] = {n: np.asarray(p, float) for n, p in zip(LM_NAMES, ln)}
        pv["projected_length_mm"] = float(Lz)
    if not np.array_equal(np.concatenate(lmk_check), lmk):
        raise RuntimeError("trace adapter: normalized landmarks differ from blocks_and_qc output")

    # ---- 조건별 관측 구성 (main.run_pipeline 과 동일)
    lmk_used = lmk.copy()
    if condition == "missing":
        lmk_used[GKC] = M["mean_LMK"][GKC]
    elif condition == "fallback":
        lmk_used[GKC] = np.nan

    # ---- 06 SSM inference
    t1 = time.time()
    if condition == "fallback":
        sys.path.insert(0, os.path.join(femur_root, "STEP37", "code"))
        import s37_common as C37
        K, gamma = 20, 0.01
        X, z, cond_no = C37.predict_f1(M, sdf[None], lmk_used[None], K, gamma)
    else:
        K, gamma = int(M["K"]), float(M["gamma"])
        X, z, cond_no = MD.predict(M, sdf[None], lmk_used[None])
    t_inf = time.time() - t1

    # ---- 07 final : main.run_pipeline 과 같은 pose / display 변환
    from phase6.phase6_ssm_xray import pose_apply
    pose = np.array(E["ck"]["E0|%s" % pid]["pose"], float)
    P = pose_apply(X[0].astype(float), pose)
    mean_shape = np.asarray(M["mean_3D"], float).reshape(-1, 3)
    P_mean = pose_apply(mean_shape, pose)               # 같은 E0 pose 로 배치한 SSM 평균 형상 (비교용)
    disp = np.linalg.norm(P - P_mean, axis=1)            # mm, 대응 정점 간 거리

    return {
        "pid": pid, "condition": condition, "W": W, "H": H, "pix_mm": PIX,
        "views": per_view, "sdf": sdf, "lmk": lmk, "lmk_used": lmk_used,
        "gkc_removed": condition == "fallback", "gkc_replaced_by_mean": condition == "missing",
        "qc": {"pass": bool(qc_ok), "checks": checks},
        "u_grid": np.asarray(CM.U_GRID, float), "v_grid": np.asarray(CM.V_GRID, float),
        "K": K, "gamma": gamma, "z": z[0], "var": np.asarray(M["var"][:K], float), "ls_condition_number": float(cond_no),
        "X": X[0], "posed": P, "posed_mean": P_mean,
        "shape_delta_mm": {"mean": float(disp.mean()), "p95": float(np.percentile(disp, 95)), "max": float(disp.max())},
        "timing_ms": {"preprocess": round(t_pre * 1000, 1), "inference": round(t_inf * 1000, 1)},
    }


# ---------------------------------------------------------------- serialization (표시용 JSON + 파일)

def _r(a, d):
    return np.round(np.asarray(a, float), d).tolist()


def write_mask_png(path: str, mask: np.ndarray) -> int:
    from PIL import Image
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("1").save(path, optimize=True)
    return os.path.getsize(path)


def trace_views_json(T: dict[str, Any], mask_url) -> list[dict[str, Any]]:
    """조건과 무관한 단계 (01–05 공통 부분). mask_url(tag) → 파일 경로/URL."""
    out = []
    for i, v in enumerate(T["views"]):
        px = np.stack([v["contour_uv_mm"][:, 0] / T["pix_mm"] + T["W"] / 2.0, T["H"] / 2.0 - v["contour_uv_mm"][:, 1] / T["pix_mm"]], 1)
        out.append({
            "tag": v["tag"], "angle_deg": v["angle_deg"],
            "threshold": {"value": v["threshold"], "bg_median": v["bg_median"], "bg_mad_scale": v["bg_mad_scale"],
                          "fixed": v["fixed_threshold"], "adaptive_used": v["adaptive_used"],
                          "raw_min": v["raw_min"], "raw_max": v["raw_max"]},
            "mask": {"url": mask_url(v["tag"]), "px": v["mask_px"], "raw_components": v["raw_components"]},
            "contour": {"n": int(len(px)), "px": _r(px, 1)},
            "sdf": {"shape": list(T["sdf"][i].shape), "min": float(T["sdf"][i].min()), "max": float(T["sdf"][i].max()),
                    "values": _r(T["sdf"][i], 4), "contour_norm": _r(v["contour_norm"], 4)},
            "landmarks": {n: {"px": _r(v["landmarks_px"][n], 1), "mm": _r(v["landmarks_mm"][n], 3), "norm": _r(v["landmarks_norm"][n], 5)} for n in LM_NAMES},
            "projected_length_mm": v["projected_length_mm"],
        })
    return out


def trace_condition_json(T: dict[str, Any]) -> dict[str, Any]:
    lu = np.asarray(T["lmk_used"], float)
    return {
        "condition": T["condition"],
        "lmk_used": [None if not np.isfinite(x) else round(float(x), 5) for x in lu],
        "gkc_removed": T["gkc_removed"], "gkc_replaced_by_mean": T["gkc_replaced_by_mean"],
        "K": T["K"], "gamma": T["gamma"],
        "z": _r(T["z"], 6), "z_over_sigma": _r(T["z"] / np.sqrt(np.maximum(T["var"], 1e-12)), 4),
        "ls_condition_number": T["ls_condition_number"],
        "shape_delta_mm": {k: round(v, 3) for k, v in T["shape_delta_mm"].items()},
    }


def trace_meta_json(T: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": T["pid"], "detector_px": [T["W"], T["H"]], "pix_mm": T["pix_mm"],
        "sdf_grid": {"u": [round(float(T["u_grid"][0]), 4), round(float(T["u_grid"][-1]), 4), int(len(T["u_grid"]))] if T["u_grid"] is not None else None,
                     "v": [round(float(T["v_grid"][0]), 4), round(float(T["v_grid"][-1]), 4), int(len(T["v_grid"]))] if T["v_grid"] is not None else None},
        "qc": T["qc"],
        "landmark_source": "3D ground-truth landmarks projected to 2D detector coordinates (research evaluation protocol) — not detected from the image",
        "threshold_rule": "max(0.02, corner-median + 3*1.4826*MAD), corners 40x40 px",
    }
