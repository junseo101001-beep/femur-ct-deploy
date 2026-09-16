# -*- coding: utf-8 -*-
"""
Femur-3D Research Demo API.

연구 코드를 **읽기 전용으로 import** 해서 최종 lock 된 파이프라인을 그대로 실행한다.

  DRR(.npy) 3장 → M1 adaptive threshold (STEP35 s35_common.processed)
               → contour (s35_common.contour_from_processed)
               → SDF/LMK 블록 (STEP33 s33_common.blocks_and_qc)
               → 잠금 B0 추론 (STEP26 s26_model.predict, K=15 γ=0.001)
               → STL

엔드포인트
  GET  /api/health              live_inference 가능 여부 · 모델 md5
  GET  /api/cases               demo_data manifest 의 케이스 목록
  POST /api/reconstruct         {case_id, condition} → 서버에서 재추론 (저장된 DRR 사용)
  POST /api/reconstruct/upload  DRR .npy 3개 업로드 → 동일 파이프라인
  GET  /api/mesh/{token}        생성된 STL

연구 파일은 절대 쓰지 않는다. 출력은 web/backend/_runs/ 에만 쓴다.
실행 : uvicorn app.main:app --port 8018   (web/backend 에서)
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys
import time
import uuid
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
FEMUR = os.path.dirname(os.path.dirname(BACKEND))          # femur-ct/
RUNS = os.path.join(BACKEND, "_runs")
DEMO = os.path.join(FEMUR, "demo_data")
os.makedirs(RUNS, exist_ok=True)

MODEL_MD5 = "83bbde9bb541db00009cabe992ed4ba8"
VIEW_TAGS = ["000.0deg", "045.0deg", "090.0deg"]

app = FastAPI(title="Femur-3D Research Demo API", version="1.0.0")
# 배포층 설정. 미설정 시 기존 동작('*') 유지.
# Render env: ALLOWED_ORIGINS=https://<pages-domain>,http://localhost:5180
_CORS_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

_ENGINE: dict[str, Any] = {"ready": False, "error": None}


def engine() -> dict[str, Any]:
    """연구 모듈 지연 로딩 (실패해도 API 는 demo 모드로 살아 있어야 한다)."""
    if _ENGINE["ready"] or _ENGINE["error"]:
        return _ENGINE
    try:
        sys.path.insert(0, os.path.join(FEMUR, "STEP35", "code"))
        sys.path.insert(0, os.path.join(FEMUR, "STEP26", "code"))
        os.chdir(FEMUR)
        import s35_common as M1C                    # STEP35 M1 전처리
        import s26_model as MD                      # 잠금 B0 추론
        C33 = M1C.C33                               # STEP33 공통 (SDF/LMK 블록 + 입력 QC)
        import hashlib
        path = os.path.join("STEP26", "final_model", "b0_final_model.npz")
        md5 = hashlib.md5(io.open(path, "rb").read()).hexdigest()
        if md5 != MODEL_MD5:
            raise RuntimeError("locked model md5 mismatch: %s" % md5)
        _ENGINE.update({
            "ready": True, "M1C": M1C, "MD": MD, "C33": C33,
            "model": MD.load_model(path), "md5": md5,
            "faces": np.load(os.path.join("phase6", "_6j_faces_ssm.npy")).astype(np.int64),
            "ck": json.load(io.open(os.path.join("STEP12", "results", "_ck.json"), encoding="utf-8")),
        })
    except Exception as e:                           # noqa: BLE001
        _ENGINE["error"] = f"{type(e).__name__}: {e}"
    return _ENGINE


def _manifest() -> dict[str, Any]:
    p = os.path.join(DEMO, "manifest.json")
    if not os.path.exists(p):
        raise HTTPException(503, "demo_data/manifest.json 이 없습니다. python web/export_demo_data.py 를 먼저 실행하세요.")
    return json.load(io.open(p, encoding="utf-8"))


def write_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> int:
    n = int(faces.shape[0])
    p = verts[faces]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(ln > 0, ln, 1.0))
    rec = np.zeros((n, 50), dtype=np.uint8)
    rec[:, :48] = np.concatenate([nrm, p[:, 0], p[:, 1], p[:, 2]], 1).astype("<f4").view(np.uint8).reshape(n, 48)
    with open(path, "wb") as f:
        f.write(b"femur-ct live inference".ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())
    return os.path.getsize(path)


def run_pipeline(pid: str, images: dict[str, np.ndarray], condition: str) -> dict[str, Any]:
    """M1 전처리 → SDF/LMK → 잠금 B0 추론. (2D landmark 는 연구 규약대로 투영값을 사용)"""
    E = engine()
    if not E["ready"]:
        raise HTTPException(503, f"live inference unavailable: {E['error']}")
    M1C, MD, C33, M = E["M1C"], E["MD"], E["C33"], E["model"]
    t0 = time.time()

    case = json.load(io.open(os.path.join(FEMUR, "drr_g", pid, "case.json"), encoding="utf-8"))
    by = {v["tag"]: v for v in case["reconstruction_input"]["views"]}
    views, thresholds = {}, []
    for tag in VIEW_TAGS:
        x, thr, med, sig = M1C.processed(images[tag], "M1")
        cont, mask, ncomp = M1C.contour_from_processed(x, thr)
        views[tag] = {"view_angle_deg": by[tag]["view_angle_deg"], "tag": tag,
                      "landmarks_2d_mm": by[tag]["landmarks_2d_mm"],
                      "contour_uv_mm": np.round(cont, 4).tolist()}
        thresholds.append({"view": tag, "threshold": float(thr), "bg_median": float(med), "bg_mad_scale": float(sig),
                           "adaptive_used": bool(thr > 0.02 + 1e-12), "mask_px": int(mask.sum()), "raw_components": int(ncomp)})
    t_pre = time.time() - t0

    sdf, lmk, checks, qc_ok = C33.blocks_and_qc(pid, views)
    if condition in ("missing", "fallback"):
        gkc = np.array([2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17])
        if condition == "missing":                       # Main 경로: 결측 열을 training 평균으로
            lmk = lmk.copy(); lmk[gkc] = M["mean_LMK"][gkc]
        else:                                            # F1: 결측 열 제거 (관측식에서 빠짐)
            lmk = lmk.copy(); lmk[gkc] = np.nan

    t1 = time.time()
    if condition == "fallback":
        sys.path.insert(0, os.path.join(FEMUR, "STEP37", "code"))
        import s37_common as C37                        # F1 추론식 (K=20, γ=0.01)
        X, z, cond_no = C37.predict_f1(M, sdf[None], lmk[None], 20, 0.01)
    else:
        X, z, cond_no = MD.predict(M, sdf[None], lmk[None])
    t_inf = time.time() - t1

    from phase6.phase6_ssm_xray import pose_apply
    pose = np.array(E["ck"]["E0|%s" % pid]["pose"], float)
    P = pose_apply(X[0].astype(float), pose)
    man = _manifest()
    c = next((cc for cc in man["cases"] if cc["pid"] == pid), None)
    if c:                                                # demo_data 와 같은 display 좌표로 맞춘다
        center = np.array(c["display_transform"]["center_mm"]); scale = c["display_transform"]["scale_mm"]
    else:
        center = (P.min(0) + P.max(0)) / 2.0; scale = float(np.linalg.norm(P - center, axis=1).max())
    V = (P - center) / scale

    token = uuid.uuid4().hex[:12]
    out = os.path.join(RUNS, f"{token}.stl")
    size = write_stl(out, V, E["faces"])
    return {
        "status": "ok", "source": "server_live", "model_md5": E["md5"], "case_id": pid, "condition": condition,
        "mesh_url": f"/api/mesh/{token}", "mesh_bytes": size, "triangles": int(E["faces"].shape[0]), "vertices": int(X.shape[1]),
        "latent": [float(v) for v in z[0]], "ls_condition_number": float(cond_no),
        "preprocessing": {"model": "M1", "rule": "max(0.02, corner-median + 3*1.4826*MAD)", "per_view": thresholds},
        "input_qc": {"pass": bool(qc_ok), "checks": checks},
        "timing_ms": {"preprocess": round(t_pre * 1000, 1), "inference": round(t_inf * 1000, 1), "total": round((time.time() - t0) * 1000, 1)},
        "note": "2D landmark 는 연구 규약대로 3D 투영값을 사용합니다 (영상 기반 검출기 아님).",
    }


# ---------------------------------------------------------------- endpoints

@app.get("/api/health")
def health() -> dict[str, Any]:
    E = engine()
    return {
        "status": "ok", "live_inference": bool(E["ready"]), "engine_error": E.get("error"),
        "model_md5": E.get("md5"), "mode": "live" if E["ready"] else "demo",
        "demo_data": os.path.exists(os.path.join(DEMO, "manifest.json")),
        "pipeline": "M1 (adaptive threshold) + locked B0 (K=15, gamma=0.001)",
        "disclaimer": "research prototype — not a clinical diagnostic system",
    }


@app.get("/api/cases")
def cases() -> dict[str, Any]:
    man = _manifest()
    return {"generated_at": man["generated_at"],
            "cases": [{"id": c["id"], "pid": c["pid"], "label": c["label"],
                       "meshes": {k: v["file"] for k, v in c["meshes"].items()},
                       "drr": {k: v["file"] for k, v in c["drr"].items()},
                       "metrics": c["metrics"]} for c in man["cases"]]}


class ReconRequest(BaseModel):
    case_id: str
    condition: str = "clean"          # clean | missing | fallback


@app.post("/api/reconstruct")
def reconstruct(req: ReconRequest) -> dict[str, Any]:
    pid = req.case_id
    d = os.path.join(FEMUR, "drr_g", pid)
    if not os.path.isdir(d):
        raise HTTPException(404, f"unknown case: {pid}")
    if req.condition not in ("clean", "missing", "fallback"):
        raise HTTPException(400, "condition must be clean | missing | fallback")
    imgs = {t: np.load(os.path.join(d, f"drr_{t}.npy")) for t in VIEW_TAGS}
    return run_pipeline(pid, imgs, req.condition)


@app.post("/api/reconstruct/upload")
async def reconstruct_upload(
    view_000: UploadFile = File(...), view_045: UploadFile = File(...), view_090: UploadFile = File(...),
    case_id: str = "Pat019", condition: str = "clean",
) -> dict[str, Any]:
    """DRR 라인적분 .npy 3장 업로드 (검출기 450×600, 0.8 mm). landmark 는 case_id 의 투영값을 사용한다."""
    imgs = {}
    for tag, up in zip(VIEW_TAGS, (view_000, view_045, view_090)):
        raw = await up.read()
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.shape != (600, 450):
            raise HTTPException(400, f"{up.filename}: expected (600, 450) float array, got {arr.shape}")
        imgs[tag] = arr.astype(np.float64)
    return run_pipeline(case_id, imgs, condition)


@app.get("/api/mesh/{token}")
def mesh(token: str) -> FileResponse:
    p = os.path.join(RUNS, f"{token}.stl")
    if not os.path.exists(p) or not token.isalnum():
        raise HTTPException(404, "mesh not found")
    return FileResponse(p, media_type="model/stl", filename=f"{token}.stl")
