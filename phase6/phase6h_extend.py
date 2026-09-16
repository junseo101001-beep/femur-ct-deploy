# -*- coding: utf-8 -*-
"""
Phase 6-H : 6-G 에서 성공한 [PC3, PC1] 기반으로 정보를 하나씩 추가한다.

바꿀 수 있는 축 (한 번에 하나만):
  - shape basis  : mode 인덱스 부분집합
  - landmark set : femurHead / greatTroch (+ kneeCenter)
  - contour 표현 : C0(현행) / C1(edge 대칭항 추가) / C2(서브픽셀 거리)
  - view 구성    : 3v60 / 4v60 / 5v60

기존 Phase 6-A~G 파일은 수정하지 않는다.
GT 는 evaluate 계열에서만.
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from phase6.phase6_ssm_xray import pose_apply, project
from phase6.phase6c_reg import surf_stats, prox_mask
from phase6.phase6g_obs import GObs, edge_points, PIX, W, H, N_OBS_CONTOUR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P6 = "phase6"
SEED = 20260912
WL, WC = 1.0, 0.25
LMSET_BASE = ("femurHead", "greatTroch")
# kneeCenter 관측 가중치. 예측기 오차(training 34명 평균 1.02mm)를 반영해 1/1.02 로 둔다.
# GT surface error 를 보고 정한 값이 아니다.
W_KNEE = 1.0 / 1.02


def log(*a):
    print(*a, flush=True)


class HObs(GObs):
    """GObs + kneeCenter 2D 관측 + 서브픽셀 거리 트리."""

    def __init__(self, pid, angles, noise=0.0, seed=None):
        GObs.__init__(self, pid, angles, noise, seed or SEED)
        d = os.path.join("drr_g", pid)
        C = json.load(io.open(os.path.join(d, "case.json"), encoding="utf-8"))
        by = {round(v["view_angle_deg"], 3): v for v in C["reconstruction_input"]["views"]}
        # kneeCenter 2D : 데이터셋 랜드마크를 같은 촬영 기하로 투영한 값.
        # DRR 생성 단계 산출물이며 reconstruction 은 이 2D 만 본다.
        from phase5_correspond import read_palp, frame_of
        raw = os.path.join("phase6", "g_raw") if pid == "Pat001" else "ssm_raw"
        lm3 = read_palp(os.path.join(raw, pid + "_palp.inp"))
        R3, t3 = _imaging_of(pid, lm3)
        kc = (lm3["kneeCenter"] @ R3.T) + t3
        self.lm["kneeCenter"] = np.array([project(kc[None, :], np.radians(a))[0]
                                          for a in angles])
        # 서브픽셀 거리: 관측 윤곽선 폴리라인을 조밀 재샘플해 KD-tree 로 둔다
        self.ctree = []
        for c in self.contour:
            p = np.vstack([c, c[:1]])
            seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
            s = np.concatenate([[0], np.cumsum(seg)])
            t = np.linspace(0, s[-1], 2400, endpoint=False)
            dense = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
            self.ctree.append(cKDTree(dense))

    def sign_of(self, i, uv):
        """마스크 기준 안/밖 부호 (+1 바깥)."""
        rc = np.stack([H / 2.0 - uv[:, 1] / PIX, uv[:, 0] / PIX + W / 2.0], 0)
        s = ndi.map_coordinates(self.sdf[i], rc, order=0, mode="nearest")
        return np.sign(s)

    def subpix_sdf(self, i, uv):
        """부호는 마스크에서, 크기는 서브픽셀 폴리라인 거리에서."""
        d, _ = self.ctree[i].query(uv)
        return self.sign_of(i, uv) * d


def _imaging_of(pid, lm):
    from phase6.phase6d_validation_drr import imaging_frame, read_stl
    raw = os.path.join("phase6", "g_raw") if pid == "Pat001" else "ssm_raw"
    tri = read_stl(os.path.join(raw, pid + ".stl"))
    return imaging_frame(lm, tri.reshape(-1, 3))


def edge_points_fixed(uv, nb=70):
    """항상 2*nb 개를 돌려준다. 빈 구간은 가장 가까운 채워진 구간 값으로 채운다.
       (phase6g_obs.edge_points 는 길이가 변해 Jacobian 에 쓸 수 없다.)"""
    v = uv[:, 1]
    e = np.linspace(v.min(), v.max(), nb + 1)
    lo = np.full(nb, np.nan); hi = np.full(nb, np.nan)
    yc = 0.5 * (e[:-1] + e[1:])
    for i in range(nb):
        m = (v >= e[i]) & (v < e[i + 1])
        if m.sum() >= 2:
            lo[i] = uv[m, 0].min(); hi[i] = uv[m, 0].max()
    idx = np.arange(nb)
    ok = ~np.isnan(lo)
    if not ok.any():
        c = uv.mean(0)
        return np.repeat(c[None, :], 2 * nb, 0)
    lo = np.interp(idx, idx[ok], lo[ok])
    hi = np.interp(idx, idx[ok], hi[ok])
    return np.vstack([np.stack([lo, yc], 1), np.stack([hi, yc], 1)])


def knee_pred(P):
    """SSM frame 정의상 kneeCenter 는 (0, -L, 0). 메시 최원위 y 로 L 을 잡는다.
       training 34 명 검증: 평균 1.02 mm, 최대 2.37 mm."""
    return np.array([0.0, P[:, 1].min(), 0.0])


class ReconH(object):
    """mode 부분집합 + landmark set + contour 표현을 고를 수 있는 shape-only 역산."""

    def __init__(self, ssm, ob, pose, mode_idx, lmset=LMSET_BASE, contour="C0",
                 lam=0.0, wC=WC, n_mesh=700):
        self.S, self.ob, self.pose = ssm, ob, np.asarray(pose, float).copy()
        self.idx = np.asarray(mode_idx, int)
        self.k = len(self.idx)
        self.B = ssm.Vt[self.idx]
        self.sd = ssm.sd[self.idx]
        self.mu, self.nv, self.gt_idx = ssm.mu, ssm.nv, ssm.gt_idx
        self.lmset, self.contour, self.lam, self.wC = tuple(lmset), contour, lam, wC
        self.sub = np.arange(0, ssm.nv, max(1, ssm.nv // n_mesh))[:n_mesh]

    def shape(self, a):
        return (self.mu + np.asarray(a) @ self.B).reshape(self.nv, 3)

    def landmarks(self, a):
        P = self.shape(a)
        d = {"femurHead": np.zeros(3), "greatTroch": P[self.gt_idx],
             "kneeCenter": knee_pred(P)}
        return np.array([d[n] for n in self.lmset])

    def _lm_w(self):
        return np.array([W_KNEE if n == "kneeCenter" else 1.0 for n in self.lmset])

    def residual(self, a, wC=None, lam=None):
        wC = self.wC if wC is None else wC
        lam = self.lam if lam is None else lam
        P = self.shape(a)
        Lw = pose_apply(self.landmarks(a), self.pose)
        Ps = pose_apply(P[self.sub], self.pose)
        w = self._lm_w()
        rl, rc = [], []
        for i, th in enumerate(self.ob.ang):
            q = project(Lw, th)
            for j, n in enumerate(self.lmset):
                rl.append(w[j] * (q[j] - self.ob.lm[n][i]))
            if wC > 0:
                uv = project(Ps, th)
                if self.contour == "C2":
                    rc.append(np.maximum(self.ob.subpix_sdf(i, uv), 0.0))
                else:
                    rc.append(np.maximum(self.ob.sample_sdf(i, uv), 0.0))
                O = self.ob.contour_fit[i]
                rc.append(np.sqrt(((O[:, None, :] - uv[None, :, :]) ** 2).sum(-1).min(1)))
                if self.contour == "C1":
                    e = edge_points_fixed(uv)
                    d, _ = self.ob.ctree[i].query(e)
                    rc.append(d)
        rl = np.concatenate(rl)
        out = [WL * rl / np.sqrt(len(rl))]
        if wC > 0:
            v = np.concatenate(rc)
            out.append(wC * v / np.sqrt(len(v)))
        if lam > 0 and self.k:
            out.append(np.sqrt(lam) * np.asarray(a) / np.maximum(self.sd, 1e-12))
        return np.concatenate(out)

    def inits(self):
        k = self.k
        out = [("A_zero", np.zeros(k))]
        rng = np.random.default_rng(SEED + k)
        for j in range(3):
            out.append(("B_small%d" % j, 0.1 * self.sd * rng.standard_normal(k)))
        for j in range(min(2, k)):
            e = np.zeros(k); e[j] = 0.5 * self.sd[j]
            out.append(("C_pc%d+" % (j + 1), e.copy()))
            out.append(("C_pc%d-" % (j + 1), -e))
        while len(out) < 8:
            out.append(("C_pad%d" % len(out), np.zeros(k)))
        for j in range(4):
            out.append(("D_rand%d" % j, self.sd * rng.uniform(-1, 1, k)))
        return out

    def fit(self, a0, iters=120):
        a = np.asarray(a0, float).copy()
        h = np.maximum(1e-2, 1e-3 * self.sd)
        r = self.residual(a); E = float(r @ r); mu = 1e-3; imp = 1e9; it = 0
        for it in range(iters):
            J = np.zeros((len(r), self.k))
            for i in range(self.k):
                q = a.copy(); q[i] += h[i]
                J[:, i] = (self.residual(q) - r) / h[i]
            A = J.T @ J; g = -J.T @ r
            ok = False
            for _ in range(6):
                try:
                    d = np.linalg.solve(A + mu * np.diag(np.diag(A))
                                        + 1e-12 * np.eye(self.k), g)
                except np.linalg.LinAlgError:
                    mu *= 10; continue
                an = a + d
                rn = self.residual(an); En = float(rn @ rn)
                if En < E:
                    imp = E - En; a, r, E = an, rn, En
                    mu = max(mu * 0.35, 1e-10); ok = True; break
                mu *= 9
            if not ok or imp < 1e-12:
                break
        r0 = self.residual(a, lam=0.0)
        return {"alpha": a, "obj_rms": float(np.sqrt((r0 ** 2).mean())),
                "iters": it + 1, "converged": bool(imp < 1e-12)}

    def jac(self, a):
        h = np.maximum(1e-2, 1e-3 * self.sd)
        r0 = self.residual(a, lam=0.0)
        J = np.zeros((len(r0), self.k))
        for i in range(self.k):
            q = a.copy(); q[i] += h[i]
            J[:, i] = (self.residual(q, lam=0.0) - r0) / h[i]
        return J

    def zstats(self, a):
        z = np.abs(np.asarray(a) / np.maximum(self.sd, 1e-12))
        return {"max": float(z.max()), "mean": float(z.mean()),
                "n_gt2": int((z > 2).sum()), "norm": float(np.linalg.norm(a))}

    def proj_metrics(self, a):
        P = self.shape(a)
        Lw = pose_apply(self.landmarks(a), self.pose)
        Ps = pose_apply(P[self.sub], self.pose)
        le, cs = [], []
        for i, th in enumerate(self.ob.ang):
            q = project(Lw, th)
            for j, n in enumerate(self.lmset):
                le.append(np.linalg.norm(q[j] - self.ob.lm[n][i]))
            uv = project(Ps, th)
            cs.append(np.maximum(self.ob.sample_sdf(i, uv), 0.0))
            O = self.ob.contour_fit[i]
            cs.append(np.sqrt(((O[:, None, :] - uv[None, :, :]) ** 2).sum(-1).min(1)))
        c = np.concatenate(cs)
        return {"lm_rms": round(float(np.sqrt(np.mean(np.array(le) ** 2))), 5),
                "contour_mean": round(float(c.mean()), 4),
                "contour_p95": round(float(np.percentile(c, 95)), 4)}

    def surfaces(self, a, tree):
        P = pose_apply(self.shape(a), self.pose)
        return {"proximal": surf_stats(P[prox_mask(P)], tree),
                "whole": surf_stats(P, tree)}
