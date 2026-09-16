# -*- coding: utf-8 -*-
"""
STEP 17 descriptor + training target (사전 등록 §3, §4). training 27명만. validation·Pat001 미사용.

관측 입력 (기존 추출 규칙 그대로)
  * 윤곽   : case.json 의 contour_uv_mm (= phase6d_validation_drr.contour_of(DRR), 240 점).
             규칙 검증 : 저장된 DRR 에 contour_of 를 다시 적용해 case.json 값과 비교.
  * 랜드마크 : femurHead, greatTroch = case.json landmarks_2d_mm (DRR 생성 단계 산출물)
             kneeCenter = phase6h_extend.HObs 와 같은 규칙 (phase5_correspond.read_palp + phase6h_extend._imaging_of
             + phase6_ssm_xray.project). 기존 파이프라인이 E0 pose 에 쓰는 2D 관측과 같은 정의.
정규화 (view 별) : 윤곽 중심 제거 → 윤곽 PCA 1축을 세로(v)로 회전, 부호는 femurHead 가 위 → v 범위 길이 L 로 나눔.
  D1 : v 방향 70 구간 중심선과 윤곽 폴리라인의 교점 좌/우 u (view 당 140).  교점이 없는 끝 구간은 이웃 선형보간.
  D2 : D1 + 정규화 좌표 랜드마크 3점 (view 당 6) → 총 420 + 18.
  D3 : view 당 10 = 5 높이(위에서 10/25/50/75/90%) 폭, 면적/L², u·v 2차 모멘트 2, u범위/v범위, 둘레/L.
  D123 : D1 ∪ 랜드마크 ∪ D3 (= D2 + D3).
Target : N27 global SSM 의 α1..α6 = alpha_of(Xn) (in-sample), σ 단위도 함께 저장.
출력 : STEP17/descriptors/descriptors_train.npz, descriptor_checks.json, STEP17/training/targets_train.npz, correspondence.json
"""
import sys, io, os, json, time, hashlib
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
sys.path.insert(0, ROOT)
for _p in ("STEP8/scripts", "STEP9/code"):
    sys.path.insert(0, os.path.join(ROOT, _p))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import s8_common as C8
from phase6.phase6k_core import NormSSM, NPZ_K
from phase6.phase6d_validation_drr import contour_of
from phase6.phase6_ssm_xray import project
from phase5_correspond import read_palp
from phase6.phase6h_extend import _imaging_of

DRR = os.path.join("STEP17", "data", "drr_train")
OD, OT = os.path.join("STEP17", "descriptors"), os.path.join("STEP17", "training")
for d in (OD, OT):
    os.makedirs(d, exist_ok=True)
TRN = list(C8.TRN)
HELD = set(C8.VAL) | {C8.TEST}
ANG = [0.0, 45.0, 90.0]
TAGS = ["000.0deg", "045.0deg", "090.0deg"]
NB = 70
HEIGHTS = [0.10, 0.25, 0.50, 0.75, 0.90]
LM = ["femurHead", "greatTroch", "kneeCenter"]


def knee_2d(pid):
    """phase6h_extend.HObs 와 같은 규칙."""
    lm3 = read_palp(os.path.join("ssm_raw", pid + "_palp.inp"))
    R3, t3 = _imaging_of(pid, lm3)
    kc = (lm3["kneeCenter"] @ R3.T) + t3
    return {a: project(kc[None, :], np.radians(a))[0] for a in ANG}


def frame(C, head):
    c = C.mean(0)
    X = C - c
    w, V = np.linalg.eigh(X.T @ X)
    e1 = V[:, np.argmax(w)]
    if (head - c) @ e1 < 0:
        e1 = -e1
    e2 = np.array([e1[1], -e1[0]])                   # 오른손 좌표 (u, v) = (e2, e1)
    R = np.stack([e2, e1], 0)
    v = X @ e1
    L = float(v.max() - v.min())
    return c, R, L


def norm(P, c, R, L):
    return ((np.asarray(P) - c) @ R.T) / L


def band_lr(Q):
    """정규화 윤곽 Q (닫힌 폴리라인) 와 v = const 선의 교점 좌/우 u."""
    v0, v1 = Q[:, 1].min(), Q[:, 1].max()
    centers = v0 + (np.arange(NB) + 0.5) * (v1 - v0) / NB
    P = np.vstack([Q, Q[:1]])
    a, b = P[:-1], P[1:]
    lo, hi = np.full(NB, np.nan), np.full(NB, np.nan)
    for i, vc in enumerate(centers):
        m = ((a[:, 1] - vc) * (b[:, 1] - vc) <= 0) & (a[:, 1] != b[:, 1])
        if m.any():
            t = (vc - a[m, 1]) / (b[m, 1] - a[m, 1])
            u = a[m, 0] + t * (b[m, 0] - a[m, 0])
            lo[i], hi[i] = u.min(), u.max()
    idx = np.arange(NB); ok = ~np.isnan(lo)
    n_empty = int((~ok).sum())
    lo = np.interp(idx, idx[ok], lo[ok]); hi = np.interp(idx, idx[ok], hi[ok])
    return np.concatenate([lo, hi]), centers, n_empty


def summary(Q, lo, hi, centers):
    v0, v1 = Q[:, 1].min(), Q[:, 1].max()
    widths = []
    for f in HEIGHTS:
        vc = v1 - f * (v1 - v0)
        j = int(np.clip(np.searchsorted(centers, vc), 0, NB - 1))
        widths.append(float(hi[j] - lo[j]))
    x, y = Q[:, 0], Q[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    per = float(np.sqrt((np.diff(np.vstack([Q, Q[:1]]), axis=0) ** 2).sum(1)).sum())
    return widths + [area, float(x.var()), float(y.var()), float((x.max() - x.min()) / (y.max() - y.min())), per]


def subject(pid):
    case = json.load(io.open(os.path.join(DRR, pid, "case.json"), encoding="utf-8"))
    views = {v["tag"]: v for v in case["reconstruction_input"]["views"]}
    kn = knee_2d(pid)
    d1, lmv, d3, chk = [], [], [], {}
    for tag, a in zip(TAGS, ANG):
        v = views[tag]
        C = np.array(v["contour_uv_mm"], float)
        img = np.load(os.path.join(DRR, pid, "drr_%s.npy" % tag))
        cr, _ = contour_of(img)
        chk[tag] = {"contour_rule_identical": bool(np.array_equal(np.round(cr, 4), C)),
                    "n_contour": int(len(C))}
        L2 = {"femurHead": np.array(v["landmarks_2d_mm"]["femurHead"]),
              "greatTroch": np.array(v["landmarks_2d_mm"]["greatTroch"]), "kneeCenter": kn[a]}
        c, R, Lz = frame(C, L2["femurHead"])
        Q = norm(C, c, R, Lz)
        lr, centers, ne = band_lr(Q)
        d1.append(lr)
        lmv.append(np.concatenate([norm(L2[n], c, R, Lz) for n in LM]))
        d3.append(summary(Q, lr[:NB], lr[NB:], centers))
        chk[tag].update({"empty_bands_interpolated": ne, "length_mm": Lz,
                         "axis_tilt_deg": float(np.degrees(np.arctan2(R[1, 0], R[1, 1]))),
                         "head_above": bool(norm(L2["femurHead"], c, R, Lz)[1] > 0),
                         "knee_below": bool(norm(L2["kneeCenter"], c, R, Lz)[1] < 0)})
    return np.concatenate(d1), np.concatenate(lmv), np.array(d3, float).ravel(), chk


def stats(M):
    return {"shape": list(M.shape), "n_nan": int(np.isnan(M).sum()), "n_inf": int(np.isinf(M).sum()),
            "min": float(np.nanmin(M)), "max": float(np.nanmax(M)), "mean": float(np.nanmean(M)),
            "col_sd_min": float(M.std(0).min()), "col_sd_max": float(M.std(0).max()),
            "n_constant_cols": int((M.std(0) < 1e-12).sum())}


if __name__ == "__main__":
    t0 = time.time()
    assert len(TRN) == 27 and not (set(TRN) & HELD)
    D1, LMK, D3, CHK = [], [], [], {}
    for pid in TRN:
        a, b, c, ch = subject(pid)
        D1.append(a); LMK.append(b); D3.append(c); CHK[pid] = ch
    D1, LMK, D3 = np.array(D1), np.array(LMK), np.array(D3)
    D2 = np.hstack([D1, LMK])
    D123 = np.hstack([D1, LMK, D3])
    sets = {"D1": D1, "D2": D2, "D3": D3, "D123": D123}
    # 결정성 : 한 번 더 계산해 비교
    again = np.array([np.concatenate(subject(p)[:3]) for p in TRN])
    determ = float(np.abs(again - D123).max())
    # target
    S = NormSSM(NPZ_K, use_pids=TRN)
    Z = np.load(NPZ_K, allow_pickle=True)
    pl = [str(p) for p in Z["pids"]]
    A = np.array([S.alpha_of(Z["Xn"][pl.index(p)])[:6] for p in TRN])
    Asig = A / S.sd[:6]
    Xn = np.stack([Z["Xn"][pl.index(p)].ravel() for p in TRN])
    np.savez_compressed(os.path.join(OD, "descriptors_train.npz"), pids=np.array(TRN), D1=D1, D2=D2, D3=D3, D123=D123,
                        landmarks_norm=LMK)
    np.savez_compressed(os.path.join(OT, "targets_train.npz"), pids=np.array(TRN), alpha=A, alpha_sigma=Asig,
                        sd=S.sd[:6], mu=S.mu, V6=S.Vt[:6], Xn=Xn)
    allchk = [c for p in CHK.values() for c in p.values()]
    checks = {"subjects": TRN, "n_subjects": len(TRN), "missing_subjects": [p for p in TRN if p not in CHK],
              "held_out_in_training": sorted(set(TRN) & HELD),
              "dims": {k: int(v.shape[1]) for k, v in sets.items()},
              "stats": {k: stats(v) for k, v in sets.items()},
              "contour_rule_identical_all": all(c["contour_rule_identical"] for c in allchk),
              "head_above_all": all(c["head_above"] for c in allchk), "knee_below_all": all(c["knee_below"] for c in allchk),
              "empty_bands_interpolated": {"max": max(c["empty_bands_interpolated"] for c in allchk),
                                           "total": sum(c["empty_bands_interpolated"] for c in allchk)},
              "axis_tilt_deg_range": [min(c["axis_tilt_deg"] for c in allchk), max(c["axis_tilt_deg"] for c in allchk)],
              "projected_length_mm_range": [min(c["length_mm"] for c in allchk), max(c["length_mm"] for c in allchk)],
              "determinism_max_abs_diff": determ, "per_subject_view": CHK,
              "target": {"alpha_sigma_mean": [float(x) for x in Asig.mean(0)], "alpha_sigma_sd": [float(x) for x in Asig.std(0)],
                         "note": "N27 SSM 의 in-sample 계수 (training 대상이 SSM 에 포함)"},
              "file_md5": {f: hashlib.md5(io.open(os.path.join(d, f), "rb").read()).hexdigest()
                           for d, f in ((OD, "descriptors_train.npz"), (OT, "targets_train.npz"))}}
    json.dump(checks, io.open(os.path.join(OD, "descriptor_checks.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"row_order": TRN, "descriptor_file": "descriptors/descriptors_train.npz", "target_file": "training/targets_train.npz",
               "descriptor_row_i == target_row_i": True,
               "blocks": {"D1": "view 0,45,90 순서, 각 [left u 70, right u 70]",
                          "landmarks_norm": "view 순서, 각 [femurHead u,v, greatTroch u,v, kneeCenter u,v]",
                          "D3": "view 순서, 각 [폭 10/25/50/75/90%, 면적, u 분산, v 분산, u범위/v범위, 둘레]"}},
              io.open(os.path.join(OT, "correspondence.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("[descriptor] 대상 %d | 차원 %s" % (len(TRN), checks["dims"]))
    for k, s in checks["stats"].items():
        print("  %-5s shape %s NaN %d Inf %d 범위 [%.3f, %.3f] 열SD [%.2e, %.2e] 상수열 %d"
              % (k, s["shape"], s["n_nan"], s["n_inf"], s["min"], s["max"], s["col_sd_min"], s["col_sd_max"], s["n_constant_cols"]))
    print("  윤곽 규칙 재적용 동일 %s | femurHead 위 %s | kneeCenter 아래 %s | 빈 구간 보간 최대 %d | 축 기울기 %s° | 투영 길이 %s mm"
          % (checks["contour_rule_identical_all"], checks["head_above_all"], checks["knee_below_all"],
             checks["empty_bands_interpolated"]["max"], np.round(checks["axis_tilt_deg_range"], 2),
             np.round(checks["projected_length_mm_range"], 1)))
    print("  결정성 재계산 최대차 %.1e | target α/σ 평균 %s SD %s" % (determ, np.round(Asig.mean(0), 3), np.round(Asig.std(0), 3)))
    print("[완료] %.0f 초" % (time.time() - t0))
