# -*- coding: utf-8 -*-
"""
Phase 6-A : SSM + landmark-only X-ray 역산  (known angle / 3 view / zero noise)

STEP 2  SSM 형상 -> 랜드마크/메시 생성 함수
STEP 3  landmark-only reconstruction
STEP 4  Pat001 GT evaluation + identifiability

데이터 분리
  - SSM basis 는 Phase 5 training subject 34 명(Pat001 제외)의 정박 대응으로 만든다.
  - reconstruction 입력: DRR 2D 랜드마크 + acquisition geometry + known view angle + SSM basis.
  - Pat001 GT mesh 는 evaluate() 안에서만 열린다. fit 계열 함수에는 전달조차 하지 않는다.
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "phase6"
SID, SOD = 1150.0, 1050.0
D1_EXTENT = 205.73          # D-1 파라메트릭 모델이 덮던 축방향 길이
KS = [0, 2, 4, 6, 8, 12, 20]


def log(*a):
    print(*a, flush=True)


# ================================================================ STEP 2
class SSM(object):
    """정박 대응으로 만든 PCA 형상 모델.  X(c) = mean + Σ c_i · PC_i"""

    def __init__(self, npz):
        Z = np.load(npz, allow_pickle=True)
        self.X = Z["X"]                       # (n, nv, 3)  training 대응점
        self.pids = [str(p) for p in Z["pids"]]
        self.gt_idx = int(Z["anchor_idx"][1])  # greatTroch 정박 인덱스
        self.head_idx = int(Z["anchor_idx"][0])
        self.femur_len = Z["femur_len"]
        n, nv, _ = self.X.shape
        self.n, self.nv = n, nv
        F = self.X.reshape(n, -1)
        self.mu = F.mean(0)
        U, S, Vt = np.linalg.svd(F - self.mu, full_matrices=False)
        self.Vt = Vt
        self.sd = np.sqrt(np.maximum((S ** 2) / (n - 1), 0))
        self.ev = self.sd ** 2 / (self.sd ** 2).sum()

    def shape(self, c):
        """계수 -> 모델 좌표계의 정점 (nv,3).  femurHead 는 원점(프레임 정의)."""
        v = self.mu.copy()
        if len(c):
            v = v + np.asarray(c) @ self.Vt[:len(c)]
        return v.reshape(self.nv, 3)

    def landmarks(self, c):
        """모델 좌표계 랜드마크.
           femurHead : 원점 — frame 정의상 항상 (0,0,0) 이라 형상 정보가 없다.
           greatTroch: 정박 인덱스 정점 — 형상에 의존한다."""
        P = self.shape(c)
        return {"femurHead": np.zeros(3), "greatTroch": P[self.gt_idx]}


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def pose_apply(P, p):
    """p = [logs, rx, ry, rz, tx, ty, tz].  모델 -> imaging frame."""
    s = np.exp(p[0])
    R = rodrigues(p[1:4])
    return s * (P @ R.T) + p[4:7]


def project(P, theta):
    """imaging frame -> 검출기 (u,v).  D-1/E-1 과 동일한 cone-beam 식."""
    c, s = np.cos(theta), np.sin(theta)
    Xr = np.stack([c * P[:, 0] + s * P[:, 2], P[:, 1],
                   -s * P[:, 0] + c * P[:, 2]], 1)
    z = Xr[:, 2] + SOD
    return np.stack([SID * Xr[:, 0] / z, SID * Xr[:, 1] / z], 1)


def triangulate(thetas, uv):
    """선형 삼각측량 (앱 triangulate 와 동일한 정규방정식)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for th, (u, v) in zip(thetas, uv):
        c, s = np.cos(th), np.sin(th)
        rows = [[-u * s - SID * c, 0.0, u * c - SID * s, -u * SOD],
                [-v * s, -SID, v * c, -v * SOD]]
        for r in rows:
            r = np.array(r)
            A += np.outer(r[:3], r[:3])
            b += r[:3] * r[3]
    return np.linalg.solve(A, b)


# ================================================================ 관측
def load_observations():
    J = json.load(io.open("ct_case_Pat001_D1.json", encoding="utf-8"))
    RI = J["reconstruction_input"]
    ang = [np.radians(v["view_angle_deg"]) for v in RI["views"]]
    obs = {}
    for name in ("femurHead", "greatTroch"):
        obs[name] = np.array([v["landmarks_2d_mm"][name] for v in RI["views"]])
    return ang, obs, RI


# ================================================================ STEP 3
class Recon(object):
    """landmark-only reconstruction.  GT 를 참조하지 않는다."""

    def __init__(self, ssm, ang, obs, k, names=("femurHead", "greatTroch"),
                 bounded=False):
        self.S, self.ang, self.obs, self.k = ssm, ang, obs, k
        self.names = names
        self.bounded = bounded
        self.np_ = k + 7

    def residual(self, p):
        c = p[7:] if self.k else []
        lm = self.S.landmarks(c)
        r = []
        for i, th in enumerate(self.ang):
            P = pose_apply(np.array([lm[n] for n in self.names]), p[:7])
            q = project(P, th)
            for j, n in enumerate(self.names):
                r.append(q[j] - self.obs[n][i])
        return np.concatenate(r)

    def rms(self, p):
        r = self.residual(p)
        return float(np.sqrt((r ** 2).mean()))

    def init(self, spin=0.0):
        """관측 2D 만 사용. 삼각측량한 두 점에 모델 두 점을 맞춘다."""
        th = triangulate(self.ang, self.obs["femurHead"])
        tg = triangulate(self.ang, self.obs["greatTroch"])
        lm = self.S.landmarks([])
        mh, mg = lm["femurHead"], lm["greatTroch"]
        dm, dt = mg - mh, tg - th
        s = np.linalg.norm(dt) / np.linalg.norm(dm)
        a, b = dm / np.linalg.norm(dm), dt / np.linalg.norm(dt)
        v = np.cross(a, b); cth = float(np.dot(a, b))
        if np.linalg.norm(v) < 1e-9:
            R0 = np.eye(3) if cth > 0 else -np.eye(3)
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R0 = np.eye(3) + K + K @ K * (1 / (1 + cth))
        Rs = rodrigues(b * spin)
        R = Rs @ R0
        rv = rotvec_of(R)
        t = th - s * (R @ mh)
        return np.concatenate([[np.log(s)], rv, t, np.zeros(self.k)])

    def jac(self, p, h=None):
        if h is None:
            h = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]
                         + [max(1e-3, 1e-3 * self.S.sd[i]) for i in range(self.k)])
        r0 = self.residual(p)
        J = np.zeros((len(r0), self.np_))
        for i in range(self.np_):
            q = p.copy(); q[i] += h[i]
            J[:, i] = (self.residual(q) - r0) / h[i]
        return J

    def clamp(self, p):
        """사전 고정한 상식 범위. 결과를 좋게 만들려고 조정한 값이 아니다.
           bounded=False 면 아무 제약도 걸지 않는다."""
        if not self.bounded:
            return p
        p[0] = np.clip(p[0], np.log(1 / 3.0), np.log(3.0))
        for i in range(self.k):
            lim = 3.0 * self.S.sd[i]
            p[7 + i] = np.clip(p[7 + i], -lim, lim)
        return p

    def fit(self, p0, iters=200):
        p = self.clamp(p0.copy())
        r = self.residual(p); E = float(r @ r); lam = 1e-3
        for it in range(iters):
            J = self.jac(p)
            A = J.T @ J; g = -J.T @ r
            ok = False
            for _ in range(12):
                try:
                    d = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(self.np_), g)
                except np.linalg.LinAlgError:
                    lam *= 10; continue
                pn = self.clamp(p + d)
                rn = self.residual(pn); En = float(rn @ rn)
                if En < E:
                    imp = E - En
                    p, r, E = pn, rn, En
                    lam = max(lam * 0.35, 1e-10); ok = True
                    break
                lam *= 9
            if not ok or (ok and imp < 1e-12):
                break
        return {"params": p, "rms": float(np.sqrt(E / len(r))), "iters": it + 1}


def rotvec_of(R):
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


# ================================================================ STEP 4 (GT)
class Evaluator(object):
    """Pat001 GT 는 여기서만 열린다."""

    def __init__(self):
        E = json.load(io.open("ct_eval_mesh_Pat001.json", encoding="utf-8"))
        self.GV = np.array(E["verts"])          # imaging frame
        self.tree = cKDTree(self.GV)
        obsj = json.load(io.open("landmark_study/Pat001/landmark_observability.json",
                                 encoding="utf-8"))
        self.lt = np.array(obsj["candidates"]["lesserTrochanter"]["position_imaging_mm"])
        c = json.load(io.open("ct_case_Pat001_D1.json", encoding="utf-8"))
        self.gt3d = c["evaluation_only"]["landmarks_3d_imaging_mm"]

    def surface(self, P):
        d = np.sort(self.tree.query(P)[0])
        return {"n": int(len(d)), "mean": round(float(d.mean()), 3),
                "median": round(float(np.median(d)), 3),
                "p95": round(float(np.percentile(d, 95)), 3),
                "max": round(float(d.max()), 3)}

    def evaluate(self, ssm, p, k):
        c = p[7:] if k else []
        P = pose_apply(ssm.shape(c), p[:7])
        lm = ssm.landmarks(c)
        L = pose_apply(np.array([lm["femurHead"], lm["greatTroch"]]), p[:7])
        e3 = {"femurHead": round(float(np.linalg.norm(L[0] - self.gt3d["femurHead"])), 4),
              "greatTroch": round(float(np.linalg.norm(L[1] - self.gt3d["greatTroch"])), 4)}
        prox = P[:, 1] > (P[:, 1].max() - D1_EXTENT)
        return {"surface_whole": self.surface(P),
                "surface_proximal_d1": self.surface(P[prox]),
                "landmark_3d_error_mm": e3,
                "femur_length_mm": round(float(P[:, 1].max() - P[:, 1].min()), 2),
                "scale": round(float(np.exp(p[0])), 5)}


# ================================================================ 실행
if __name__ == "__main__":
    t0 = time.time()
    S = SSM(os.path.join(OUT, "corr_train_anchored.npz"))
    ang, obs, RI = load_observations()
    log("[STEP 1] SSM basis: training %d 명, 정점 %d, greatTroch 정박 인덱스 %d"
        % (S.n, S.nv, S.gt_idx))
    log("  설명분산 PC1~5: " + " ".join("%.1f%%" % (100 * x) for x in S.ev[:5]))
    log("  view 각도 (known): %s" % [round(np.degrees(a), 1) for a in ang])
    log("")
    log("[STEP 2/3] landmark-only reconstruction (femurHead + greatTroch)")
    log("  관측 정보량: 랜드마크 2 개 x 3 view x 2 = 12 잔차,")
    log("  그러나 독립 3D 좌표는 2 점 x 3 = 6 개뿐이다.")
    log("  미지수 = k(형상) + 7(scale 1 + 회전 3 + 이동 3)")
    log("")

    EV = Evaluator()
    RES, MS = {}, {}
    log("  k   미지수  잔차   rank null  cond        RMS(mm)    surf(근위) med/p95/max")
    for k in KS:
        R = Recon(S, ang, obs, k)
        f = R.fit(R.init())
        J = R.jac(f["params"])
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int((sv > sv[0] * 1e-6).sum())
        nul = R.np_ - rank
        cond = float(sv[0] / max(sv[-1], 1e-300))
        ev = EV.evaluate(S, f["params"], k)
        RES["k%d" % k] = {"n_params": R.np_, "n_res": len(R.residual(f["params"])),
                          "rms_mm": round(f["rms"], 6), "iters": f["iters"],
                          "singular": [float("%.4g" % x) for x in sv],
                          "rank": rank, "null_dim": nul, "cond": float("%.4g" % cond),
                          "params": [round(float(x), 5) for x in f["params"]],
                          "coef_z": [round(float(f["params"][7 + i] / S.sd[i]), 3)
                                     for i in range(k)],
                          **ev}
        sp = ev["surface_proximal_d1"]
        log("  %-3d %5d %5d %5d %4d %11.4g %10.4f   %6.3f %6.3f %6.3f"
            % (k, R.np_, RES["k%d" % k]["n_res"], rank, nul, cond, f["rms"],
               sp["median"], sp["p95"], sp["max"]))

    # ---------------- multi-start ----------------
    log("")
    log("[STEP 3b] multi-start (초기 spin 각을 0~2π 로 12 등분)")
    log("  k    RMS 범위                  scale 범위      계수 범위(첫 성분)")
    for bounded in (False, True):
      tag = "bounded" if bounded else "free"
      log("  --- %s (%s) ---" % (tag, "scale 1/3~3배, 계수 ±3σ" if bounded else "제약 없음"))
      for k in KS:
        R = Recon(S, ang, obs, k, bounded=bounded)
        rows = []
        for spin in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            f = R.fit(R.init(spin=float(spin)))
            rows.append(f)
        rms = np.array([f["rms"] for f in rows])
        sc = np.array([np.exp(f["params"][0]) for f in rows])
        c1 = np.array([f["params"][7] for f in rows]) if k else np.zeros(len(rows))
        surf = [EV.evaluate(S, f["params"], k)["surface_proximal_d1"]["median"]
                for f in rows]
        zmax = max(abs(f["params"][7 + i]) / S.sd[i]
                   for f in rows for i in range(k)) if k else 0.0
        MS["%s_k%d" % (tag, k)] = {"n_start": len(rows), "max_abs_z": float(zmax),
                         "rms_min": float(rms.min()), "rms_max": float(rms.max()),
                         "scale_min": float(sc.min()), "scale_max": float(sc.max()),
                         "c1_min": float(c1.min()), "c1_max": float(c1.max()),
                         "surf_median_min": float(min(surf)),
                         "surf_median_max": float(max(surf))}
        log("  %-3d  %.4g ~ %.4g   %.4f ~ %.4f   %.1f ~ %.1f   (surf med %.2f ~ %.2f)"
            % (k, rms.min(), rms.max(), sc.min(), sc.max(), c1.min(), c1.max(),
               min(surf), max(surf)))

    json.dump({"phase": "6-A", "condition": "known angle / 3 view / zero noise / landmark-only",
               "landmarks_used": ["femurHead", "greatTroch"],
               "lesserTrochanter_excluded_reason":
                   "SSM 템플릿 해상도(4911 점)에서 Part C 의 소전자 기하 규칙을 적용하면 "
                   "기록된 관측에서 51.0 mm 떨어진 점이 나온다(조밀 원본 18915 점에서는 "
                   "1.85 mm). 규칙이 해상도에 의존하고 subject 간에도 22/34 개의 서로 다른 "
                   "대응 인덱스로 흩어진다. 따라서 이 SSM 으로 소전자를 예측할 수 없다.",
               "information_budget": {
                   "n_residuals": 12, "independent_3d_coordinates": 6,
                   "note": "femurHead 는 SSM frame 정의상 항상 원점이므로 형상 정보를 담지 "
                           "않는다. 형상에 의존하는 관측은 greatTroch 3 좌표뿐이다."},
               "ssm": {"n_training": S.n, "n_vertices": S.nv,
                       "explained_variance": [round(float(x), 5) for x in S.ev[:20]],
                       "greatTroch_anchor_index": S.gt_idx},
               "results": RES, "multistart": MS},
              io.open(os.path.join(OUT, "phase6_ssm_xray_results.json"), "w",
                      encoding="utf-8"), indent=2, ensure_ascii=False)
    log("")
    log("완료 %.0f 초 -> %s" % (time.time() - t0,
                                os.path.join(OUT, "phase6_ssm_xray_results.json")))
