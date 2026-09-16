# -*- coding: utf-8 -*-
"""
Phase 5 / STEP 3 : mesh 정합과 대응(correspondence) 확보.

  ssm_raw/PatNNN.stl + PatNNN_palp.inp
      -> 랜드마크 기반 해부학적 좌표계로 강체 정렬
      -> 템플릿 기반 non-rigid ICP (Laplacian 정규화)
      -> 모든 subject 가 같은 인덱스의 정점 집합을 갖게 된다

Pat001 은 이 단계에 들어오지 않는다. training subject 만 처리한다.
Pat001 은 phase5_ssm.py 의 evaluation 단계에서만 같은 방식으로 대응을 만든다.

출력 : ssm_work/corr_train.npz
"""
import sys, io, os, csv, json, struct, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.linalg as spl

RAW = "ssm_raw"
WORK = "ssm_work"
if not os.path.isdir(WORK):
    os.makedirs(WORK)
N_TEMPLATE = 5000          # 템플릿 정점 수 (사전 고정)
KNN = 8                    # Laplacian 그래프 이웃 수
ITERS = 14
LAM0, LAM1 = 3000.0, 3.0   # 정규화 세기 (기하급수 감소)
REJECT0, REJECT1 = 60.0, 8.0   # 대응 거부 임계 (mm)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 입출력
def read_stl(path):
    b = io.open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    v = a[:, 12:48].copy().view(np.float32).reshape(n * 3, 3).astype(np.float64)
    return v, n


def dedup(v, q=1e-4):
    k = np.round(v / q).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return v[np.sort(idx)]


def read_palp(path):
    d = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        if "=" not in line:
            continue
        k, val = line.split("=", 1)
        d[k.strip()] = float(val.strip())
    out = {}
    for nm in ("femurHead", "headDir", "greatTroch", "kneeCenter"):
        out[nm] = np.array([d[nm + "X"], d[nm + "Y"], d[nm + "Z"]])
    return out


def frame_of(lm):
    """랜드마크 4 점으로 해부학적 좌표계를 만든다.
       원점 = femurHead,  y = 무릎->골두(장축),  x = 대전자의 축 수직 성분."""
    H, G, K = lm["femurHead"], lm["greatTroch"], lm["kneeCenter"]
    y = H - K
    L = np.linalg.norm(y)
    y = y / L
    x = G - H
    x = x - np.dot(x, y) * y
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    R = np.stack([x, y, z], 1)          # 열이 축
    return R, H, L


def to_frame(v, R, H):
    return (v - H) @ R


# ---------------------------------------------------------------- 로드
def load_subject(pid, raw=RAW):
    v, ntri = read_stl(os.path.join(raw, pid + ".stl"))
    v = dedup(v)
    lm = read_palp(os.path.join(raw, pid + "_palp.inp"))
    R, H, L = frame_of(lm)
    return {"pid": pid, "V": to_frame(v, R, H), "ntri": ntri,
            "femur_len": L, "R": R, "H": H,
            "lm": {k: to_frame(x[None, :], R, H)[0] for k, x in lm.items()}}


# ---------------------------------------------------------------- 템플릿
def voxel_downsample(V, target):
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    s = (np.prod(ext) / target) ** (1.0 / 3.0)
    for _ in range(40):
        key = np.floor((V - lo) / s).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        if abs(len(idx) - target) / target < 0.05:
            break
        s *= (len(idx) / float(target)) ** (1.0 / 3.0)
    return V[np.sort(idx)]


def knn_laplacian(P, k=KNN):
    t = cKDTree(P)
    d, j = t.query(P, k=k + 1)
    n = len(P)
    rows = np.repeat(np.arange(n), k)
    cols = j[:, 1:].ravel()
    w = np.ones(len(rows))
    A = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(deg) - A


# ---------------------------------------------------------------- non-rigid ICP
def fit_template(T, L, target_V, iters=ITERS):
    """T: 템플릿 정점(n,3), L: 템플릿 Laplacian, target_V: 대상 정점.
       Laplacian 정규화 non-rigid ICP. 반환: 변형된 템플릿, 최종 잔차 통계."""
    tree = cKDTree(target_V)
    X = T.copy()
    n = len(T)
    LtL = (L.T @ L).tocsc()
    for it in range(iters):
        f = it / max(iters - 1, 1)
        lam = LAM0 * (LAM1 / LAM0) ** f
        rej = REJECT0 * (REJECT1 / REJECT0) ** f
        d, j = tree.query(X)
        w = (d < rej).astype(np.float64)
        if w.sum() < 0.2 * n:          # 너무 적으면 전부 사용
            w[:] = 1.0
        C = target_V[j]
        W = sp.diags(w)
        A = (W + lam * LtL).tocsc()
        solve = spl.factorized(A)
        for c in range(3):
            X[:, c] = solve(w * C[:, c])
    d, _ = tree.query(X)
    return X, {"mean": float(d.mean()), "median": float(np.median(d)),
               "p95": float(np.percentile(d, 95)), "max": float(d.max())}


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    meta = list(csv.reader(io.open("HFValid_Metadata.csv", encoding="utf-8-sig"),
                           delimiter=";"))
    hdr = meta[0]
    iP, iF = hdr.index("PatientID"), hdr.index("FractureStatus")
    frac = {r[iP]: r[iF] for r in meta[1:]}

    cand = json.load(io.open("ssm_training_subjects.json",
                             encoding="utf-8"))["training_candidates"]
    have = [p for p in cand if os.path.exists(os.path.join(RAW, p + ".stl"))]
    # 사전 규칙: 골절 subject 는 제외한다 (형상이 정상 대퇴골이 아니다). GT 무관.
    excl_frac = [p for p in have if frac.get(p) != "0"]
    train = [p for p in have if frac.get(p) == "0"]
    log("다운로드된 후보 %d, 골절 제외 %d -> training %d 명"
        % (len(have), len(excl_frac), len(train)))
    if excl_frac:
        log("  골절로 제외: %s" % ", ".join(excl_frac))

    t0 = time.time()
    subs = []
    for p in train:
        s = load_subject(p)
        subs.append(s)
        log("  로드 %s  정점 %6d  삼각형 %6d  대퇴골장 %.1f mm"
            % (p, len(s["V"]), s["ntri"], s["femur_len"]))
    log("로드 %.0f 초" % (time.time() - t0))

    lens = np.array([s["femur_len"] for s in subs])
    # 템플릿: 대퇴골장이 중앙값에 가장 가까운 subject. 결정적 규칙, GT 무관.
    ti = int(np.argmin(np.abs(lens - np.median(lens))))
    log("")
    log("템플릿 subject = %s (대퇴골장 %.1f mm, 중앙값 %.1f mm)"
        % (subs[ti]["pid"], lens[ti], np.median(lens)))

    T = voxel_downsample(subs[ti]["V"], N_TEMPLATE)
    log("템플릿 정점 %d 개로 다운샘플" % len(T))
    Lap = knn_laplacian(T)

    t0 = time.time()
    X = np.zeros((len(subs), len(T), 3))
    resid = []
    for i, s in enumerate(subs):
        Xi, st = fit_template(T, Lap, s["V"])
        X[i] = Xi
        resid.append(st)
        log("  [%2d/%d] %s  대응잔차 mean %.3f  median %.3f  p95 %.3f  max %.3f mm"
            % (i + 1, len(subs), s["pid"], st["mean"], st["median"],
               st["p95"], st["max"]))
    log("대응 %.0f 초" % (time.time() - t0))

    np.savez_compressed(os.path.join(WORK, "corr_train.npz"),
                        X=X, T=T, pids=np.array([s["pid"] for s in subs]),
                        femur_len=lens, template_pid=subs[ti]["pid"])
    json.dump({"n_training": len(subs), "training_subjects": train,
               "excluded_fracture": excl_frac,
               "template_subject": subs[ti]["pid"],
               "template_rule": "training subject 중 대퇴골장이 중앙값에 가장 가까운 subject. "
                                "결정적 규칙이며 Pat001 이나 GT 를 참조하지 않는다.",
               "n_template_vertices": int(len(T)),
               "alignment": "랜드마크 4 점(femurHead, headDir, greatTroch, kneeCenter)으로 "
                            "만든 해부학적 좌표계. 원점 femurHead, y = kneeCenter->femurHead, "
                            "x = greatTroch 의 축 수직 성분.",
               "correspondence_method": "템플릿 기반 non-rigid ICP. Laplacian(kNN k=%d) "
                                        "정규화, %d 회 반복, lambda %g->%g, 대응 거부 "
                                        "임계 %g->%g mm." % (KNN, ITERS, LAM0, LAM1,
                                                             REJECT0, REJECT1),
               "correspondence_residual_mm": resid,
               "pat001_used": False},
              io.open(os.path.join(WORK, "correspondence_report.json"), "w",
                      encoding="utf-8"), indent=2, ensure_ascii=False)
    log("")
    log("저장: %s" % os.path.join(WORK, "corr_train.npz"))
