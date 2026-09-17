# -*- coding: utf-8 -*-
"""
DEMO MODE 용 reconstruction process 데이터 생성 (read-only).

web/backend/app/trace.py 의 adapter 로 잠금 파이프라인을 1회씩 실행해
실제 중간 결과를 demo_data/process/<pid>/ 에 저장하고 frontend public 으로 복사한다.

  demo_data/process/<pid>/
    trace.json          meta · views(01–05: threshold 값, contour px, SDF 3×112×40, landmark px/정규화)
                        · conditions{clean, missing, fallback}(06: K, γ, z, z/σ, shape Δ)
    mask_000.png …      03 실제 mask (1-bit PNG)
    mean_shape.stl      06 SSM 평균 형상 (같은 E0 pose · display 변환)
  07 최종 mesh 는 기존 demo_data/meshes/<pid>_recon_*.stl 을 그대로 참조한다
  (verify_trace.py 에서 trace 결과와 정점 차이 0 확인).

연구 파일은 읽기만 한다. 실행 : python web/export_process_data.py   (femur-ct 에서)
"""
import io
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web", "backend"))
from app import main                 # noqa: E402
from app import trace as TR          # noqa: E402

OUT = os.path.join(ROOT, "demo_data", "process")
FRONT = os.path.join(ROOT, "web", "frontend", "public", "demo-data", "process")
MESH_KEY = {"clean": "recon_clean", "missing": "recon_missing_main", "fallback": "recon_fallback_f1"}


def run():
    t0 = time.time()
    E = main.engine()
    if not E["ready"]:
        raise SystemExit("engine not ready: %s" % E["error"])
    man = json.load(io.open(os.path.join(main.DEMO, "manifest.json"), encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    index = []
    for c in man["cases"]:
        pid = c["pid"]
        d = os.path.join(OUT, pid)
        os.makedirs(d, exist_ok=True)
        imgs = {t: np.load(os.path.join(main.FEMUR, "drr_g", pid, "drr_%s.npy" % t)) for t in main.VIEW_TAGS}
        center = np.array(c["display_transform"]["center_mm"]); scale = c["display_transform"]["scale_mm"]

        conds, views, meta = {}, None, None
        for cond in ("clean", "missing", "fallback"):
            T = TR.trace_pipeline(E, main.FEMUR, pid, imgs, cond)
            if views is None:                                # 01–05 는 조건과 무관 (같은 전처리)
                for v in T["views"]:
                    TR.write_mask_png(os.path.join(d, "mask_%s.png" % v["tag"][:3]), v["mask"])
                views = TR.trace_views_json(T, lambda tag: "process/%s/mask_%s.png" % (pid, tag[:3]))
                meta = TR.trace_meta_json(T)
                main.write_stl(os.path.join(d, "mean_shape.stl"), (T["posed_mean"] - center) / scale, E["faces"])
            cj = TR.trace_condition_json(T)
            cj["final_mesh"] = "meshes/%s_%s.stl" % (pid, MESH_KEY[cond])
            conds[cond] = cj
        doc = {"case_id": pid, "mode": "precomputed", "model_md5": E["md5"],
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta, "views": views,
               "conditions": conds, "files": {"mean_shape_mesh": "process/%s/mean_shape.stl" % pid}}
        json.dump(doc, io.open(os.path.join(d, "trace.json"), "w", encoding="utf-8"), separators=(",", ":"), ensure_ascii=False)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        index.append({"pid": pid, "bytes": size})
        print("[%s] %.1f KB" % (pid, size / 1024))

    json.dump({"cases": index}, io.open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"), indent=1)
    if os.path.isdir(FRONT):
        shutil.rmtree(FRONT)
    shutil.copytree(OUT, FRONT)
    total = sum(x["bytes"] for x in index)
    print("done: %.2f MB → %s | %.1f s" % (total / 1e6, FRONT, time.time() - t0))


if __name__ == "__main__":
    run()
