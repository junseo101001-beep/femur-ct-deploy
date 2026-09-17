# -*- coding: utf-8 -*-
"""
Trace adapter 동일성 검증 (read-only).

같은 입력(drr_g 저장 DRR)에 대해
  (1) 기존 /api/reconstruct 경로  main.run_pipeline            → latent, STL
  (2) 새 trace 경로               trace.trace_pipeline          → latent, 추정 형상, STL
  (3) 연구 저장 prediction        STEP37/results/predictions.npz (A clean · B missing-main · C F1, zA/zB/zC)
  (4) 연구 저장 입력 블록          STEP35/results/inputs/M1/clean/<pid>/b0_input.npz (sdf, lmk)
  (5) 웹 demo mesh                demo_data/meshes/<pid>_recon_*.stl

를 비교한다. 결과는 web/backend/_runs/trace_verification.json 에만 쓴다.
실행 : python verify_trace.py   (web/backend 에서)
"""
import io
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from app import main                     # noqa: E402
from app import trace as TR              # noqa: E402

COND_KEY = {"clean": "A", "missing": "B", "fallback": "C"}
MESH_KEY = {"clean": "recon_clean", "missing": "recon_missing_main", "fallback": "recon_fallback_f1"}


def stl_vertices(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84:84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(rec[:, 12:48].tobytes(), dtype="<f4").reshape(n, 3, 3), b[84:]


def main_():
    E = main.engine()
    if not E["ready"]:
        raise SystemExit("engine not ready: %s" % E["error"])
    femur = main.FEMUR
    man = json.load(io.open(os.path.join(main.DEMO, "manifest.json"), encoding="utf-8"))
    pred = np.load(os.path.join(femur, "STEP37", "results", "predictions.npz"))
    pids37 = [str(p) for p in pred["pids"]]
    rows, all_ok = [], True
    t0 = time.time()
    for c in man["cases"]:
        pid = c["pid"]
        imgs = {t: np.load(os.path.join(femur, "drr_g", pid, "drr_%s.npy" % t)) for t in main.VIEW_TAGS}
        center = np.array(c["display_transform"]["center_mm"]); scale = c["display_transform"]["scale_mm"]
        for cond in ("clean", "missing", "fallback"):
            # (1) 기존 경로
            R = main.run_pipeline(pid, imgs, cond)
            v_orig, body_orig = stl_vertices(os.path.join(main.RUNS, R["mesh_url"].rsplit("/", 1)[1] + ".stl"))
            # (2) trace 경로
            T = TR.trace_pipeline(E, femur, pid, imgs, cond)
            tmp = os.path.join(main.RUNS, "_verify_%s_%s.stl" % (pid, cond))
            main.write_stl(tmp, (T["posed"] - center) / scale, E["faces"])
            v_tr, body_tr = stl_vertices(tmp)
            os.remove(tmp)
            # (3) 연구 prediction
            i = pids37.index(pid)
            X_res = pred[COND_KEY[cond]][i]; z_res = pred["z" + COND_KEY[cond]][i]
            # (5) demo mesh
            v_demo, _ = stl_vertices(os.path.join(main.DEMO, "meshes", "%s_%s.stl" % (pid, MESH_KEY[cond])))

            r = {
                "pid": pid, "condition": cond,
                "latent_equal_to_reconstruct_api": bool(np.array_equal(np.array(R["latent"]), T["z"])),
                "stl_bytes_equal_to_reconstruct_api": body_orig == body_tr,
                "vertex_max_abs_vs_reconstruct_api": float(np.abs(v_orig - v_tr).max()),
                "shape_max_abs_vs_research_prediction_mm": float(np.abs(T["X"] - X_res).max()),
                "latent_max_abs_vs_research_prediction": float(np.abs(T["z"] - z_res).max()),
                "display_vertex_max_abs_vs_demo_mesh": float(np.abs(v_tr - v_demo).max()),
                "K": T["K"], "gamma": T["gamma"],
            }
            if cond == "clean":
                ref = np.load(os.path.join(femur, "STEP35", "results", "inputs", "M1", "clean", pid, "b0_input.npz"))
                r["sdf_max_abs_vs_research_input"] = float(np.abs(T["sdf"] - ref["sdf"]).max())
                r["lmk_max_abs_vs_research_input"] = float(np.abs(T["lmk"] - ref["lmk"]).max())
            ok = (r["latent_equal_to_reconstruct_api"] and r["stl_bytes_equal_to_reconstruct_api"]
                  and r["shape_max_abs_vs_research_prediction_mm"] < 1e-6 and r["latent_max_abs_vs_research_prediction"] < 1e-9
                  and r["display_vertex_max_abs_vs_demo_mesh"] < 1e-6
                  and r.get("sdf_max_abs_vs_research_input", 0.0) < 1e-9 and r.get("lmk_max_abs_vs_research_input", 0.0) < 1e-9)
            r["pass"] = bool(ok); all_ok &= ok
            rows.append(r)
            print("%s %-8s pass=%s  latentEq=%s stlEq=%s  |X-res|=%.2e |z-res|=%.2e |demo|=%.2e%s" % (
                pid, cond, ok, r["latent_equal_to_reconstruct_api"], r["stl_bytes_equal_to_reconstruct_api"],
                r["shape_max_abs_vs_research_prediction_mm"], r["latent_max_abs_vs_research_prediction"], r["display_vertex_max_abs_vs_demo_mesh"],
                ("  |sdf|=%.2e |lmk|=%.2e" % (r["sdf_max_abs_vs_research_input"], r["lmk_max_abs_vs_research_input"])) if cond == "clean" else ""))
    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model_md5": E["md5"], "all_pass": bool(all_ok),
           "tolerances": {"research_prediction_mm": 1e-6, "latent": 1e-9, "demo_mesh_display_units": 1e-6, "sdf_lmk": 1e-9},
           "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, io.open(os.path.join(main.RUNS, "trace_verification.json"), "w", encoding="utf-8"), indent=1)
    print("ALL PASS" if all_ok else "MISMATCH FOUND", "| %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main_()
