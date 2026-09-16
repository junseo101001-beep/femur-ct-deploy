# -*- coding: utf-8 -*-
"""
STEP38 최종 lock 상태의 연구 산출물 → 웹 demo_data 변환기 (연구 파일은 읽기 전용).

생성물
  demo_data/meshes/<pid>_{recon_clean,recon_missing_main,recon_fallback_f1,gt}.stl
  demo_data/meshes/<pid>_{noise50_m0,noise50_m1}.stl        (SNR50 비교용, 1 대상)
  demo_data/drr/<pid>_{000,045,090}.png                      (clean)
  demo_data/drr/<pid>_noise50_{000,045,090}.png              (SNR50, 1 대상)
  demo_data/research.json                                    (모든 지표 · lock 정보)
  demo_data/manifest.json                                    (케이스 목록 · 파일 경로)

정렬 규약
  recon = pose_apply(Xn_hat, E0 pose)  → imaging frame (평가와 동일한 변환)
  GT    = ssm_raw STL → imaging_frame(R, t) → 같은 imaging frame
  두 메시에 **같은** display 변환 (GT bbox 중심 / 최대 반경) 을 적용하므로 화면 상 정합이 평가 좌표계와 같다.

실행 : python web/export_demo_data.py          (femur-ct 루트에서)
"""
import os, sys, io, json, csv, struct, shutil, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # femur-ct/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "STEP26", "code"))
os.chdir(ROOT)

from phase6.phase6d_validation_drr import imaging_frame, read_stl, read_palp   # 연구 코드 그대로
from phase6.phase6_ssm_xray import pose_apply                                   # 평가와 같은 pose 적용
import s26_model as MD                                                          # 잠금 B0 추론

OUT = os.path.join(ROOT, "demo_data")
FRONT_PUBLIC = os.path.join(ROOT, "web", "frontend", "public", "demo-data")
VAL = ["Pat019", "Pat031", "Pat042", "Pat060", "Pat080", "Pat095"]
TAGS = [("000.0", "000", 0), ("045.0", "045", 45), ("090.0", "090", 90)]
NOISE_CASE = "Pat019"          # SNR50 시각 비교용 (STEP34/35 저장 입력 사용)
GT_STRIDE = 6                  # GT STL 삼각형 서브샘플 (웹 전송용)


def log(*a):
    print(*a, flush=True)


def write_stl_binary(path, verts, faces, header=b"femur-ct research demo"):
    n = int(faces.shape[0])
    p = verts[faces]                                    # (n,3,3)
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    nrm = np.cross(u, v)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(ln > 0, ln, 1.0))
    rec = np.zeros((n, 50), dtype=np.uint8)
    rec[:, :48] = np.concatenate([nrm, p[:, 0], p[:, 1], p[:, 2]], axis=1).astype("<f4").view(np.uint8).reshape(n, 48)
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())
    return os.path.getsize(path)


def png_from_drr(arr, path):
    """DRR(선적분) → 8bit PNG. min-max 정규화 후 v 축 뒤집기 (화면 좌표)."""
    from PIL import Image
    a = np.asarray(arr, float)
    lo, hi = float(a.min()), float(a.max())
    img = ((a - lo) / max(hi - lo, 1e-12) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path, optimize=True)
    return {"file": os.path.relpath(path, OUT).replace("\\", "/"), "shape": list(a.shape), "raw_min": lo, "raw_max": hi}


def display_transform(gt_verts):
    c = (gt_verts.min(0) + gt_verts.max(0)) / 2.0
    s = float(np.linalg.norm(gt_verts - c, axis=1).max())
    return c, s


def apply_display(v, c, s):
    return (np.asarray(v, float) - c) / s


def jload(p):
    return json.load(io.open(p, encoding="utf-8"))


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "drr"), exist_ok=True)

    lock = jload(os.path.join("STEP38", "results", "final_lock.json"))
    v26 = jload(os.path.join("STEP26", "validation", "validation_results.json"))
    v37 = jload(os.path.join("STEP37", "results", "validation_f1_results.json"))
    v39 = jload(os.path.join("STEP39", "results", "validation_geom_results.json"))
    ck = jload(os.path.join("STEP12", "results", "_ck.json"))
    M = MD.load_model(os.path.join("STEP26", "final_model", "b0_final_model.npz"))
    faces = np.load(os.path.join("phase6", "_6j_faces_ssm.npy")).astype(np.int64)
    pred37 = np.load(os.path.join("STEP37", "results", "predictions.npz"))
    pids37 = [str(x) for x in pred37["pids"]]
    log("faces", faces.shape, "| STEP37 preds", {k: pred37[k].shape for k in ("A", "B", "C")})

    cases = []
    for ci, pid in enumerate(VAL):
        idx = pids37.index(pid)
        pose = np.array(ck["E0|%s" % pid]["pose"], float)

        # --- GT : ssm_raw STL → imaging frame (평가와 같은 변환)
        tri = read_stl(os.path.join("ssm_raw", pid + ".stl"))
        lm3 = read_palp(os.path.join("ssm_raw", pid + "_palp.inp"))
        R, t = imaging_frame(lm3, tri.reshape(-1, 3))
        gt_tri = (tri.reshape(-1, 3) @ R.T + t).reshape(-1, 3, 3)
        c, s = display_transform(gt_tri.reshape(-1, 3))
        sub = gt_tri[::GT_STRIDE]
        gv = apply_display(sub.reshape(-1, 3), c, s)
        gf = np.arange(len(gv), dtype=np.int64).reshape(-1, 3)
        gt_path = os.path.join(OUT, "meshes", "%s_gt.stl" % pid)
        gt_bytes = write_stl_binary(gt_path, gv, gf)

        meshes = {"gt": {"file": "meshes/%s_gt.stl" % pid, "triangles": int(len(gf)), "bytes": gt_bytes,
                         "note": "expert CT segmentation STL, %d/%d triangles subsampled for web" % (len(gf), len(gt_tri))}}
        # --- reconstruction 3종 (STEP37 A/B/C), 같은 display 변환 → GT 와 정합
        for key, label in (("A", "recon_clean"), ("B", "recon_missing_main"), ("C", "recon_fallback_f1")):
            P = pose_apply(pred37[key][idx].astype(float), pose)
            vv = apply_display(P, c, s)
            path = os.path.join(OUT, "meshes", "%s_%s.stl" % (pid, label))
            b = write_stl_binary(path, vv, faces)
            meshes[label] = {"file": "meshes/%s_%s.stl" % (pid, label), "triangles": int(len(faces)), "bytes": b}

        # --- clean DRR 3 view
        drr = {}
        for tag, short, deg in TAGS:
            a = np.load(os.path.join("drr_g", pid, "drr_%sdeg.npy" % tag))
            info = png_from_drr(a, os.path.join(OUT, "drr", "%s_%s.png" % (pid, short)))
            info["angle_deg"] = deg
            drr[short] = info

        ps37 = v37["per_subject"][pid]
        ps26 = v26["per_subject"][pid]
        cases.append({
            "id": "case%02d" % (ci + 1), "pid": pid, "label": "Case %02d" % (ci + 1),
            "meshes": meshes, "drr": drr,
            "metrics": {
                "clean_m1": {k: ps37["A_clean_M1"][k] for k in ("sym", "p95", "cov5", "vol_err_pct", "alpha_rmse_sigma", "rot_deg", "trans_mm")},
                "missing_main": {k: ps37["B_missing_main"][k] for k in ("sym", "p95", "cov5", "vol_err_pct", "alpha_rmse_sigma")},
                "fallback_f1": {k: ps37["C_missing_F1"][k] for k in ("sym", "p95", "cov5", "vol_err_pct", "alpha_rmse_sigma")},
                "baseline_e0": {"sym": ps26["baseline"]["sym"], "p95": ps26["baseline"]["p95"], "cov5": ps26["baseline"]["cov5"],
                                "vol_err_pct": ps26["baseline"]["vol_err_pct"], "alpha_rmse_sigma": ps26["baseline"]["alpha_rmse_g6"]},
                "delta_f1_minus_control": ps37["d_sym_C_minus_B"],
                "delta_b0_minus_e0": ps26["delta_sym"],
            },
            "display_transform": {"center_mm": c.tolist(), "scale_mm": s},
            "pose_source": "STEP12 _ck.json E0|%s (평가와 동일)" % pid,
        })
        log("[%s] gt %d tris (%.1f KB) | recon %d tris | drr 3" % (pid, len(gf), gt_bytes / 1024, len(faces)))

    # --- SNR50 시각 비교 (M0 붕괴 vs M1) : 저장된 입력에 잠금 모델을 그대로 적용
    noise = None
    m0_in = os.path.join("STEP34", "results", "drr", "snr050_r0", NOISE_CASE, "b0_input.npz")
    m1_in = os.path.join("STEP35", "results", "inputs", "M1", "snr050_r0", NOISE_CASE, "b0_input.npz")
    if os.path.exists(m0_in) and os.path.exists(m1_in):
        ci = [c_["pid"] for c_ in cases].index(NOISE_CASE)
        cc = np.array(cases[ci]["display_transform"]["center_mm"]); ss = cases[ci]["display_transform"]["scale_mm"]
        pose = np.array(ck["E0|%s" % NOISE_CASE]["pose"], float)
        nm = {}
        for label, path in (("noise50_m0", m0_in), ("noise50_m1", m1_in)):
            z = np.load(path)
            X, _, _ = MD.predict(M, z["sdf"][None], z["lmk"][None])
            vv = apply_display(pose_apply(X[0].astype(float), pose), cc, ss)
            out = os.path.join(OUT, "meshes", "%s_%s.stl" % (NOISE_CASE, label))
            b = write_stl_binary(out, vv, faces)
            nm[label] = {"file": "meshes/%s_%s.stl" % (NOISE_CASE, label), "triangles": int(len(faces)), "bytes": b}
            cases[ci]["meshes"][label] = nm[label]
        nd = {}
        for tag, short, deg in TAGS:
            a = np.load(os.path.join("STEP34", "results", "drr", "snr050_r0", NOISE_CASE, "drr_%sdeg.npy" % tag))
            info = png_from_drr(a, os.path.join(OUT, "drr", "%s_noise50_%s.png" % (NOISE_CASE, short)))
            info["angle_deg"] = deg
            nd[short] = info
            cases[ci]["drr"]["noise50_" + short] = info
        noise = {"case_pid": NOISE_CASE, "snr": 50, "sigma": 0.0156171,
                 "m0_input": m0_in.replace("\\", "/"), "m1_input": m1_in.replace("\\", "/"),
                 "meshes": nm, "drr": nd,
                 "note": "STEP34(M0)·STEP35(M1) 에 저장된 입력에 잠금 B0 를 그대로 적용해 메시만 생성"}
        log("[noise] SNR50 M0/M1 메시·DRR 생성 (%s)" % NOISE_CASE)

    # --- 전체 지표 (FINAL_RESULTS.csv = STEP38 확정치)
    rows = list(csv.DictReader(io.open(os.path.join("STEP38", "FINAL_RESULTS.csv"), encoding="utf-8")))
    num = lambda x: float(x) if x not in (None, "", "NA") else None
    final_rows = [{"category": r["category"], "condition": r["condition"], "model": r["model"],
                   "sym": num(r["sym_mean_mm"]), "p95": num(r["p95_mm"]), "cov5": num(r["cov5_pct"]),
                   "abs_vol": num(r["abs_vol_err_pct"]), "source": r["source"]} for r in rows]

    research = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_of_truth": ["STEP38/results/final_lock.json", "STEP38/FINAL_RESULTS.csv", "STEP26/validation/validation_results.json",
                            "STEP36/results/per_subject.csv", "STEP37/results/validation_f1_results.json", "STEP39/results/validation_geom_results.json"],
        "lock": lock,
        "final_rows": final_rows,
        "validation_cohort": VAL,
        "clean": {
            "e0": {"sym": v26["summary"]["baseline"]["sym"], "p95": v26["summary"]["baseline"]["p95"], "cov5": v26["summary"]["baseline"]["cov5"],
                   "abs_vol": v26["summary"]["baseline"]["abs_vol_err_pct"], "alpha": v26["summary"]["baseline"]["alpha_rmse_g6"]},
            "b0": {"sym": v26["summary"]["B0"]["sym"], "p95": v26["summary"]["B0"]["p95"], "cov5": v26["summary"]["B0"]["cov5"],
                   "abs_vol": v26["summary"]["B0"]["abs_vol_err_pct"], "alpha": v26["summary"]["B0"]["alpha_rmse_g6"]},
            "delta_b0_e0": v26["delta_sym_vs_1.4985"], "n_better": v26["n_subjects_better"],
            "per_subject": {p: {"e0": v26["per_subject"][p]["baseline"]["sym"], "b0": v26["per_subject"][p]["B0"]["sym"],
                                "delta": v26["per_subject"][p]["delta_sym"], "improved": v26["per_subject"][p]["delta_sym"] < 0,
                                "p95_b0": v26["per_subject"][p]["B0"]["p95"], "p95_e0": v26["per_subject"][p]["baseline"]["p95"],
                                "cov5_b0": v26["per_subject"][p]["B0"]["cov5"], "vol_b0": v26["per_subject"][p]["B0"]["vol_err_pct"]} for p in VAL},
        },
        "missing": {
            "control": v37["summary"]["B_missing_main"], "f1": v37["summary"]["C_missing_F1"], "clean_m1": v37["summary"]["A_clean_M1"],
            "delta_f1_control": v37["delta_C_minus_B"], "criteria": v37["criteria"], "verdict": v37["verdict"],
            "per_subject": {p: {"clean": v37["per_subject"][p]["A_clean_M1"], "control": v37["per_subject"][p]["B_missing_main"],
                                "f1": v37["per_subject"][p]["C_missing_F1"], "delta": v37["per_subject"][p]["d_sym_C_minus_B"]} for p in VAL},
            "geom_probe_step39": {"sym": v39["summary"]["F1_FALLBACK_GEOM"]["sym"]["mean"], "f0_sym": v39["summary"]["F0_FALLBACK_BASE"]["sym"]["mean"],
                                  "verdict": v39["verdict"], "adopted": v39["fallback_adopted"], "adopted_fallback": v39["adopted_fallback"]},
        },
        "noise_demo": noise,
    }
    json.dump(research, io.open(os.path.join(OUT, "research.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    manifest = {"generated_at": research["generated_at"], "cases": cases, "noise_demo": noise,
                "mesh_convention": {"units": "display-normalized (GT bbox center, max radius = 1)",
                                    "alignment": "recon = pose_apply(Xn_hat, E0 pose); GT = imaging_frame(STL); 두 메시 모두 동일 display 변환 → 평가 좌표계와 동일 정합",
                                    "recon_topology": "SSM 대응 4911 vertices / %d triangles" % len(faces)}}
    json.dump(manifest, io.open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # --- frontend public 복사
    if os.path.isdir(FRONT_PUBLIC):
        shutil.rmtree(FRONT_PUBLIC)
    shutil.copytree(OUT, FRONT_PUBLIC)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    log("완료 : %s (%.1f MB) → %s | %.0f 초" % (OUT, total / 1e6, FRONT_PUBLIC, time.time() - t0))
