# -*- coding: utf-8 -*-
"""
Phase 6-B : SSM + multi-view X-ray contour 역산
            (known angle / 3 view / zero noise / landmark + contour / lambda=0)

STEP 1  SSM basis 로드 검증
STEP 2  SSM mesh -> projection -> predicted contour 항
STEP 3  단일 조건 실행
STEP 4  k = 4/6/8/12/20
STEP 5  wC sweep
STEP 6  multi-start identifiability
STEP 7  Pat001 GT evaluation

데이터 분리
  - SSM basis : Phase 6 정박 대응(Pat001 제외 training 34 명).
  - reconstruction 입력 : DRR 2D 랜드마크 + DRR 에서 추출한 contour/mask
                          + acquisition geometry + known view angle + SSM basis.
  - Pat001 GT 는 class Evaluator 안에서만 열린다.
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from phase6.phase6_ssm_xray import (SSM, rodrigues, rotvec_of, pose_apply,
                                    project, triangulate, SID, SOD, D1_EXTENT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P6 = "phase6"
TAGS = ["000", "030", "060"]
KS = [4, 6, 8, 12, 20]
WCS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
WC_DEFAULT = 0.25          # STEP 3/4 기준값. 결과를 보기 전에 정했다.
THR = 0.02                 # E-1 과 동일 (뼈 경로 0.5~0.8 mm 근거)
N_MESH_CONTOUR = 700       # contour 항에 쓰는 메시 점 수 (결정적 stride)
N_OBS_CONTOUR = 120        # coverage 항에 쓰는 관측 윤곽선 점 수 (240 중 결정적 stride)


def log(*a):
    print(*a, flush=True)


# ================================================================ 관측
class Observation(object):
    """DRR 과 그 파생물만 읽는다. GT mesh / CT / GT 3D 랜드마크를 읽지 않는다."""

    def __init__(self):
        J = json.load(io.open("ct_case_Pat001_D1.json", encoding="utf-8"))
        RI = J["reconstruction_input"]
        self.ang = [np.radians(v["view_angle_deg"]) for v in RI["views"]]
        self.lm = {n: np.array([v["landmarks_2d_mm"][n] for v in RI["views"]])
                   for n in ("femurHead", "greatTroch")}
        C = json.load(io.open("contour/Pat001/contours.json", encoding="utf-8"))
        d = C["detector"]
        self.PIX, self.W, self.H = d["pixel_spacing_mm"], d["width_px"], d["height_px"]
        self.contour = [np.array(C["contours"][t]["points_uv_mm"]) for t in TAGS]
        self.contour_fit = [c[::max(1, len(c) // N_OBS_CONTOUR)][:N_OBS_CONTOUR]
                            for c in self.contour]
        self.sdf = []
        for t in TAGS:
            a = np.load(os.path.join("drr", "Pat001", "drr_%sdeg.npy" % t)).astype(float)
            m = a > THR
            m = ndi.binary_closing(m, np.ones((3, 3)))
            m = ndi.binary_fill_holes(m)
            lab, nl = ndi.label(m)
            if nl:
                sz = ndi.sum(m, lab, range(1, nl + 1))
                m = lab == (int(np.argmax(sz)) + 1)
            # 부호 있는 거리장 (mm). 바깥이 양수.
            din = ndi.distance_transform_edt(m) * self.PIX
            dout = ndi.distance_transform_edt(~m) * self.PIX
            self.sdf.append(dout - din)
        self.files_read = (["ct_case_Pat001_D1.json", "contour/Pat001/contours.json"]
                           + ["drr/Pat001/drr_%sdeg.npy" % t for t in TAGS])

    def uv_to_rc(self, uv):
        return np.stack([self.H / 2.0 - uv[:, 1] / self.PIX,
                         uv[:, 0] / self.PIX + self.W / 2.0], 0)

    def sample_sdf(self, i, uv):
        rc = self.uv_to_rc(uv)
        return ndi.map_coordinates(self.sdf[i], rc, order=1, mode="nearest")


# ================================================================ STEP 2 잔차
class ReconC(object):
    """landmark + contour reconstruction.

    contour 잔차는 예측 실루엣 곡선을 뽑지 않고 두 개의 연속 항으로 만든다.
      containment : 투영된 모델 점이 관측 실루엣 영역 밖에 있으면 그 거리만큼 벌한다.
                    r = max(SDF_obs(proj(x)), 0)
      coverage    : 관측 윤곽선의 각 점에서 가장 가까운 투영 모델 점까지 거리.
    두 항이 함께 예측 실루엣을 관측 실루엣에 붙인다. 실루엣 추출이 없으므로
    파라미터에 대해 연속이다. (E-1 의 ring 극점 방식은 위상이 있는 lofted mesh
    전용이라 점군 SSM 에 이식되지 않는다.)
    """

    LMN = ("femurHead", "greatTroch")

    def __init__(self, ssm, ob, k, wC, bounded=False):
        self.S, self.ob, self.k, self.wC = ssm, ob, k, wC
        self.bounded = bounded
        self.np_ = k + 7
        self.sub = np.arange(0, ssm.nv, max(1, ssm.nv // N_MESH_CONTOUR))[:N_MESH_CONTOUR]

    # ---------- 예측 ----------
    def _posed(self, p):
        c = p[7:] if self.k else []
        P = self.S.shape(c)
        lm = self.S.landmarks(c)
        return pose_apply(P, p[:7]), pose_apply(
            np.array([lm[n] for n in self.LMN]), p[:7])

    def residual(self, p):
        Pw, Lw = self._posed(p)
        rl, rc = [], []
        Ps = Pw[self.sub]
        for i, th in enumerate(self.ang_list()):
            q = project(Lw, th)
            for j, n in enumerate(self.LMN):
                rl.append(q[j] - self.ob.lm[n][i])
            if self.wC > 0:
                uv = project(Ps, th)
                rc.append(np.maximum(self.ob.sample_sdf(i, uv), 0.0))
                O = self.ob.contour_fit[i]
                d2 = ((O[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
                rc.append(np.sqrt(d2.min(1)))
        rl = np.concatenate(rl)
        out = rl / np.sqrt(len(rl))
        if self.wC > 0:
            rcv = np.concatenate(rc)
            out = np.concatenate([out, self.wC * rcv / np.sqrt(len(rcv))])
        return out

    def ang_list(self):
        return self.ob.ang

    # ---------- 평가용 지표 (목적함수와 별개) ----------
    def metrics(self, p):
        Pw, Lw = self._posed(p)
        lm_e, cont = [], []
        for i, th in enumerate(self.ang_list()):
            q = project(Lw, th)
            for j, n in enumerate(self.LMN):
                lm_e.append(np.linalg.norm(q[j] - self.ob.lm[n][i]))
            uv = project(Pw, th)
            # E-1 비교용: 예측 외곽선(v 구간별 u 최소/최대) -> 관측 윤곽선 chamfer
            pred = outline_of(uv)
            O = self.ob.contour[i]
            d = np.sqrt(((pred[:, None, :] - O[None, :, :]) ** 2).sum(-1)).min(1)
            cont.append(d)
        c = np.concatenate(cont)
        return {"lm_rms_mm": round(float(np.sqrt((np.array(lm_e) ** 2).mean())), 5),
                "contour_pred_to_obs": {"mean": round(float(c.mean()), 3),
                                        "median": round(float(np.median(c)), 3),
                                        "p95": round(float(np.percentile(c, 95)), 3),
                                        "max": round(float(c.max()), 3)}}

    # ---------- 최적화 ----------
    def clamp(self, p):
        if not self.bounded:
            return p
        p[0] = np.clip(p[0], np.log(1 / 3.0), np.log(3.0))
        for i in range(self.k):
            lim = 3.0 * self.S.sd[i]
            p[7 + i] = np.clip(p[7 + i], -lim, lim)
        return p

    def init(self, spin=0.0, cinit=None):
        th = triangulate(self.ob.ang, self.ob.lm["femurHead"])
        tg = triangulate(self.ob.ang, self.ob.lm["greatTroch"])
        lm = self.S.landmarks([])
        mh, mg = lm["femurHead"], lm["greatTroch"]
        dm, dt = mg - mh, tg - th
        s = np.linalg.norm(dt) / np.linalg.norm(dm)
        a, b = dm / np.linalg.norm(dm), dt / np.linalg.norm(dt)
        v = np.cross(a, b); ct = float(np.dot(a, b))
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R0 = np.eye(3) + K + K @ K * (1 / (1 + ct)) if np.linalg.norm(v) > 1e-9 else np.eye(3)
        R = rodrigues(b * spin) @ R0
        c0 = np.zeros(self.k) if cinit is None else np.asarray(cinit, float)
        return np.concatenate([[np.log(s)], rotvec_of(R), th - s * (R @ mh), c0])

    def jac(self, p):
        h = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]
                     + [max(1e-2, 1e-3 * self.S.sd[i]) for i in range(self.k)])
        r0 = self.residual(p)
        J = np.zeros((len(r0), self.np_))
        for i in range(self.np_):
            q = p.copy(); q[i] += h[i]
            J[:, i] = (self.residual(q) - r0) / h[i]
        return J

    def fit(self, p0, iters=60):
        p = self.clamp(p0.copy())
        r = self.residual(p); E = float(r @ r); lam = 1e-3; imp = 1e9
        for it in range(iters):
            J = self.jac(p)
            A = J.T @ J; g = -J.T @ r
            ok = False
            for _ in range(6):
                try:
                    d = np.linalg.solve(A + lam * np.diag(np.diag(A))
                                        + 1e-12 * np.eye(self.np_), g)
                except np.linalg.LinAlgError:
                    lam *= 10; continue
                pn = self.clamp(p + d)
                rn = self.residual(pn); En = float(rn @ rn)
                if En < E:
                    imp = E - En; p, r, E = pn, rn, En
                    lam = max(lam * 0.35, 1e-10); ok = True; break
                lam *= 9
            if not ok or imp < 1e-12:
                break
        return {"params": p, "obj_rms": float(np.sqrt(E / len(r))), "iters": it + 1}


def outline_of(uv, nb=70):
    """투영 점군의 외곽선: v 를 구간으로 나눠 각 구간의 u 최소/최대. 평가 전용."""
    v = uv[:, 1]
    e = np.linspace(v.min(), v.max(), nb + 1)
    pts = []
    for i in range(nb):
        m = (v >= e[i]) & (v < e[i + 1])
        if m.sum() < 2:
            continue
        yc = 0.5 * (e[i] + e[i + 1])
        pts.append([uv[m, 0].min(), yc]); pts.append([uv[m, 0].max(), yc])
    return np.array(pts)


# ================================================================ STEP 7 GT
class Evaluator(object):
    """Pat001 GT 는 여기서만 열린다."""

    def __init__(self):
        E = json.load(io.open("ct_eval_mesh_Pat001.json", encoding="utf-8"))
        self.tree = cKDTree(np.array(E["verts"]))
        c = json.load(io.open("ct_case_Pat001_D1.json", encoding="utf-8"))
        self.gt3d = c["evaluation_only"]["landmarks_3d_imaging_mm"]

    def surf(self, P):
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
        prox = P[:, 1] > (P[:, 1].max() - D1_EXTENT)
        return {"surface_whole": self.surf(P),
                "surface_proximal_d1": self.surf(P[prox]),
                "landmark_3d_error_mm": {
                    "femurHead": round(float(np.linalg.norm(L[0] - self.gt3d["femurHead"])), 4),
                    "greatTroch": round(float(np.linalg.norm(L[1] - self.gt3d["greatTroch"])), 4)},
                "femur_length_mm": round(float(P[:, 1].max() - P[:, 1].min()), 2),
                "scale": round(float(np.exp(p[0])), 5)}


# ================================================================ 실행
if __name__ == "__main__":
    t0 = time.time()
    S = SSM(os.path.join(P6, "corr_train_anchored.npz"))
    ob = Observation()
    EV = Evaluator()
    log("[STEP 1] SSM training %d 명, 정점 %d, greatTroch 정박 %d"
        % (S.n, S.nv, S.gt_idx))
    log("  설명분산 PC1~5: " + " ".join("%.1f%%" % (100 * x) for x in S.ev[:5]))
    log("  관측: 랜드마크 2 x 3 view, contour %d 점/view, threshold %.3f"
        % (len(ob.contour[0]), THR))
    log("  contour 항에 쓰는 메시 점 %d 개" % N_MESH_CONTOUR)

    RES = {}
    N_SPIN = 8
    SPINS = np.linspace(0, 2 * np.pi, N_SPIN, endpoint=False)
    log("")
    log("[STEP 3/4/5] k x wC grid.  각 조건마다 초기 spin %d 등분으로 다중 시작하고" % N_SPIN)
    log("  목적함수가 가장 낮은 해를 그 조건의 결과로 삼는다(GT 를 보지 않는 선택 규칙).")
    log("  k   wC     obj      lmRMS   cont.med  rank null  surf med   p95     max   | "
        "spin 간 surf med 범위")
    for k in KS:
        for wC in WCS:
            R = ReconC(S, ob, k, wC)
            rows = [R.fit(R.init(spin=float(sp))) for sp in SPINS]
            objs = np.array([f["obj_rms"] for f in rows])
            evs = [EV.evaluate(S, f["params"], k) for f in rows]
            sms = np.array([e["surface_proximal_d1"]["median"] for e in evs])
            b = int(np.argmin(objs))
            f, e = rows[b], evs[b]
            J = R.jac(f["params"])
            sv = np.linalg.svd(J, compute_uv=False)
            rank = int((sv > sv[0] * 1e-6).sum()); nul = R.np_ - rank
            m = R.metrics(f["params"])
            sp_ = e["surface_proximal_d1"]
            RES["k%d_wC%g" % (k, wC)] = {
                "k": k, "wC": wC, "n_params": R.np_,
                "n_res": int(len(R.residual(f["params"]))),
                "obj_rms": round(f["obj_rms"], 6), "iters": f["iters"],
                "best_spin_deg": round(float(np.degrees(SPINS[b])), 1),
                "rank": rank, "null_dim": nul,
                "cond": float("%.4g" % (sv[0] / max(sv[-1], 1e-300))),
                "singular": [float("%.4g" % x) for x in sv],
                "coef_z": [round(float(f["params"][7 + i] / S.sd[i]), 3) for i in range(k)],
                "multistart": {"n_start": int(N_SPIN),
                               "obj_min": float(objs.min()), "obj_max": float(objs.max()),
                               "surf_med_min": float(sms.min()),
                               "surf_med_max": float(sms.max()),
                               "surf_med_sd": float(sms.std(ddof=1)),
                               "surf_med_at_best_obj": float(sms[b]),
                               "obj_vs_surf": [[round(float(o), 5), round(float(x), 3)]
                                               for o, x in zip(objs, sms)]},
                **m, **e}
            log("  %-3d %-5g %8.4f %8.4f %8.3f %5d %4d %8.3f %7.3f %7.3f   | %.2f ~ %.2f"
                % (k, wC, f["obj_rms"], m["lm_rms_mm"],
                   m["contour_pred_to_obs"]["median"], rank, nul,
                   sp_["median"], sp_["p95"], sp_["max"], sms.min(), sms.max()))

    log("")
    log("[STEP 6] 심화 multi-start : spin 8 등분 x 계수 초기값 3 종 (0, +1sigma, -1sigma)")
    MS = {}
    for k in (8, 20):
        for wC in (0.0, WC_DEFAULT, 1.0):
            R = ReconC(S, ob, k, wC)
            rows = []
            for sp in SPINS:
                for tag in (None, "plus", "minus"):
                    ci = None if tag is None else (
                        (1.0 if tag == "plus" else -1.0) * S.sd[:k])
                    rows.append(R.fit(R.init(spin=float(sp), cinit=ci)))
            objs = np.array([f["obj_rms"] for f in rows])
            sms = np.array([EV.evaluate(S, f["params"], k)["surface_proximal_d1"]["median"]
                            for f in rows])
            b = int(np.argmin(objs))
            zmax = max(abs(f["params"][7 + i]) / S.sd[i] for f in rows for i in range(k))
            MS["k%d_wC%g" % (k, wC)] = {
                "n_start": len(rows), "obj_min": float(objs.min()),
                "obj_max": float(objs.max()), "surf_med_min": float(sms.min()),
                "surf_med_max": float(sms.max()), "surf_med_sd": float(sms.std(ddof=1)),
                "surf_med_at_best_obj": float(sms[b]), "max_abs_z": float(zmax),
                "spearman_obj_vs_surf": float(np.corrcoef(
                    np.argsort(np.argsort(objs)), np.argsort(np.argsort(sms)))[0, 1])}
            log("  k=%-3d wC=%-5g  obj %.4f~%.4f  surf med %.2f~%.2f (sd %.2f) "
                " best-obj 해 %.2f  max|z| %.1f  순위상관 %.2f"
                % (k, wC, objs.min(), objs.max(), sms.min(), sms.max(),
                   sms.std(ddof=1), sms[b], zmax,
                   MS["k%d_wC%g" % (k, wC)]["spearman_obj_vs_surf"]))

    json.dump({"phase": "6-B",
               "condition": "known angle / 3 view (0,30,60) / zero noise / "
                            "landmark + contour / lambda = 0",
               "contour_residual_definition":
                   "예측 실루엣 곡선을 추출하지 않는다. 두 개의 연속 항을 쓴다. "
                   "(1) containment: 투영된 모델 점이 관측 실루엣 영역 밖이면 "
                   "SDF 거리만큼 벌한다. (2) coverage: 관측 윤곽선 각 점에서 가장 가까운 "
                   "투영 모델 점까지의 거리. E-1 의 ring 극점 실루엣 방식은 위상이 있는 "
                   "lofted mesh 전용이라 점군 SSM 에 이식되지 않아 대체했다. "
                   "보고용 contour 지표(contour_pred_to_obs)는 별도로 v 구간별 u 최소/최대 "
                   "외곽선을 뽑아 관측 윤곽선까지 chamfer 로 잰 값이며 목적함수와 무관하다.",
               "weighting": "R = [ wL/sqrt(NL) * r_landmark , wC/sqrt(NC) * r_contour ], wL=1",
               "wc_default_for_step34": WC_DEFAULT,
               "observation_files": ob.files_read,
               "idealized_contour_caveat":
                   "Phase 4-B 의 DRR 은 전문가 STL 마스크 바깥의 HU 를 -1000 으로 만든 뒤 "
                   "생성되었다(phase4b.py:244). 따라서 여기서 쓴 contour 는 실제 임상 X-ray "
                   "에서 독립 annotation 한 윤곽선이 아니라 이상적 상한이다.",
               "ssm": {"source": "phase6/corr_train_anchored.npz",
                       "n_training": S.n, "n_vertices": S.nv,
                       "explained_variance": [round(float(x), 5) for x in S.ev[:20]],
                       "pat001_in_training": False},
               "grid": RES, "multistart": MS},
              io.open(os.path.join(P6, "phase6b_results.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    log("")
    log("완료 %.0f 초 -> %s" % (time.time() - t0, os.path.join(P6, "phase6b_results.json")))
