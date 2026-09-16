# -*- coding: utf-8 -*-
"""
Phase 6-F : pose 를 Phase 6-E 결과로 고정하고 SSM PCA 계수만 추정한다.

Stage 1 : Phase 6-E 의 검증된 pose (init B / optimizer A, meanShape) 를 그대로 재사용.
          재계산하지 않는다 -> 6-E 와 정확히 일치함이 보장된다.
Stage 2 : pose 고정. 미지수는 alpha_1..alpha_k 뿐.
          optimizer 는 rotation / translation / scale 을 건드릴 수 없다.

GT 는 evaluate 계열에서만. reconstruction 경로는 GT 를 읽지 않는다.
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from phase6.phase6_ssm_xray import pose_apply, project
from phase6.phase6c_reg import surf_stats, prox_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P6 = "phase6"
N_MESH_CONTOUR = 700
N_OBS_CONTOUR = 120
WL, WC = 1.0, 0.25            # 6-B~E 와 동일. 사전 고정.
KS = [2, 4, 6, 8, 12, 20]
LAMBDAS = [0.0, 0.01, 0.05, 0.1]
SEED = 20260910               # 결정적 재현용


def log(*a):
    print(*a, flush=True)


def pose_from_6e(pid):
    """Phase 6-E (init B / opt A) 의 pose 를 그대로 가져온다. GT 아님."""
    R = json.load(io.open(os.path.join(P6, "phase6e_results.json"), encoding="utf-8"))
    A = dict(R["validation"]); A.update(R["test"])
    for v in A.values():
        if v["subject"] == pid and v["init"] == "B" and v["opt"] == "A":
            b = v["best"]
            return np.concatenate([[np.log(b["scale"])], b["rotvec"], b["trans"]])
    raise KeyError(pid)


class ReconF(object):
    """pose 고정, alpha 만 추정."""

    LMN = ("femurHead", "greatTroch")

    def __init__(self, ssm, ob, pose, k, lam=0.0, wC=WC):
        self.S, self.ob, self.k, self.lam, self.wC = ssm, ob, k, lam, wC
        self.pose = np.asarray(pose, float).copy()      # 절대 바뀌지 않는다
        self.mu = ssm.mu
        self.B = ssm.Vt[:k]                              # (k, 3nv)
        self.sd = ssm.sd[:k]
        self.nv = ssm.nv
        self.gt_idx = ssm.gt_idx
        self.sub = np.arange(0, ssm.nv, max(1, ssm.nv // N_MESH_CONTOUR))[:N_MESH_CONTOUR]

    # ---------------- 예측 ----------------
    def shape(self, a):
        return (self.mu + np.asarray(a) @ self.B).reshape(self.nv, 3)

    def landmarks(self, a):
        P = self.shape(a)
        return np.array([np.zeros(3), P[self.gt_idx]])

    def residual(self, a, wC=None, lam=None):
        wC = self.wC if wC is None else wC
        lam = self.lam if lam is None else lam
        P = self.shape(a)
        Lw = pose_apply(self.landmarks(a), self.pose)
        Ps = pose_apply(P[self.sub], self.pose)
        rl, rc = [], []
        for i, th in enumerate(self.ob.ang):
            q = project(Lw, th)
            for j, n in enumerate(self.LMN):
                rl.append(q[j] - self.ob.lm[n][i])
            if wC > 0:
                uv = project(Ps, th)
                rc.append(np.maximum(self.ob.sample_sdf(i, uv), 0.0))
                O = self.ob.contour_fit[i]
                rc.append(np.sqrt(((O[:, None, :] - uv[None, :, :]) ** 2).sum(-1).min(1)))
        rl = np.concatenate(rl)
        out = [WL * rl / np.sqrt(len(rl))]
        if wC > 0:
            v = np.concatenate(rc)
            out.append(wC * v / np.sqrt(len(v)))
        if lam > 0 and self.k:
            out.append(np.sqrt(lam) * np.asarray(a) / np.maximum(self.sd, 1e-12))
        return np.concatenate(out)

    def obj(self, a, with_reg=False):
        r = self.residual(a, lam=(self.lam if with_reg else 0.0))
        return float(np.sqrt((r ** 2).mean()))

    # ---------------- 초기화 (GT 미사용) ----------------
    def inits(self):
        """12 개. 전부 결정적. 관측/모델 prior 만 사용."""
        k = self.k
        out = [("A_zero", np.zeros(k))]
        rng = np.random.default_rng(SEED + k)
        for j in range(3):                                   # B: 작은 섭동
            out.append(("B_small%d" % j, 0.1 * self.sd * rng.standard_normal(k)))
        for j in range(min(2, k)):                           # C: PC 방향 +-0.5 sigma
            e = np.zeros(k); e[j] = 0.5 * self.sd[j]
            out.append(("C_pc%d+" % (j + 1), e.copy()))
            out.append(("C_pc%d-" % (j + 1), -e))
        while len(out) < 8:
            out.append(("C_pad%d" % len(out), np.zeros(k)))
        for j in range(4):                                   # D: +-1 sigma 무작위
            out.append(("D_rand%d" % j, self.sd * rng.uniform(-1, 1, k)))
        return out

    # ---------------- 최적화 (alpha 만) ----------------
    def fit(self, a0, iters=120):
        a = np.asarray(a0, float).copy()
        h = np.maximum(1e-2, 1e-3 * self.sd)
        r = self.residual(a); E = float(r @ r); mu = 1e-3; imp = 1e9
        it = 0
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
                "obj_with_reg": float(np.sqrt((r ** 2).mean())),
                "iters": it + 1, "converged": bool(imp < 1e-12)}

    # ---------------- 진단 ----------------
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
        return {"max": float(z.max()) if self.k else 0.0,
                "mean": float(z.mean()) if self.k else 0.0,
                "n_gt2": int((z > 2).sum()), "n_gt3": int((z > 3).sum()),
                "norm": float(np.linalg.norm(np.asarray(a)))}

    def proj_metrics(self, a):
        P = self.shape(a)
        Lw = pose_apply(self.landmarks(a), self.pose)
        Ps = pose_apply(P[self.sub], self.pose)
        le, cs = [], []
        for i, th in enumerate(self.ob.ang):
            q = project(Lw, th)
            for j, n in enumerate(self.LMN):
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
        """GT tree 를 인자로 받는다 -> 호출자가 evaluation 임을 명시한다."""
        P = pose_apply(self.shape(a), self.pose)
        return {"proximal": surf_stats(P[prox_mask(P)], tree),
                "whole": surf_stats(P, tree)}
