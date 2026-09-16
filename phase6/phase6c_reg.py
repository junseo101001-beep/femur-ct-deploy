# -*- coding: utf-8 -*-
"""
Phase 6-C : regularization + 최적화 전략으로 다중 해 문제를 줄일 수 있는가.

관측(Pat001)은 Phase 6-B 와 완전히 동일하다. 바꾸는 것은
  (1) SSM coefficient regularization  E_reg = lambda * sum (c_i/sigma_i)^2
  (2) 최적화 전략 (single-stage vs coarse-to-fine)
뿐이다.

데이터 분리
  - lambda 는 validation subject 로만 고른다. Pat001 GT 를 절대 보지 않는다.
  - validation subject 는 training 34 명을 결정적 규칙으로 27/7 로 나눈 뒤의 7 명.
    validation 용 PCA basis 는 27 명으로만 만든다.
  - Pat001 GT 는 class Evaluator 안에서만 열린다.

한계 (보고서에 명시)
  - validation subject 에는 DRR 이 없다(Phase 4-B 는 Pat001 만 생성). 그래서 validation
    관측은 그 subject 의 메시를 같은 촬영 기하로 투영해 합성한다. Pat001 관측은
    DRR 에서 추출한 것 그대로다. 두 채널의 차이는 아래 obs_channel_check 로 정량화한다.
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from phase5_correspond import read_stl, dedup, read_palp, frame_of, to_frame
from phase6.phase6_ssm_xray import rodrigues, rotvec_of, pose_apply, project, triangulate
from phase6.phase6b_ssm_contour import Observation, outline_of, TAGS, THR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P6 = "phase6"
SID, SOD = 1150.0, 1050.0
D1_EXTENT = 205.73
PIX, W, H = 0.8, 450, 600
VIEWS = [0.0, 30.0, 60.0]
N_MESH_CONTOUR = 700
N_OBS_CONTOUR = 120
N_CONTOUR_PTS = 240

# ---- 사전 고정 설정 -------------------------------------------------------
LAMBDAS = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
WC_VALID = 0.25                      # lambda 선택에 쓰는 대표 wC
WC_FINAL = [0.01, 0.25, 0.5]         # Pat001 최종에서 볼 wC
K_VALID = 8                          # lambda 선택에 쓰는 대표 k
K_FINAL = [8, 20]
N_SPIN_VALID = 3
N_SPIN_FINAL = 8
CINIT_FINAL = (None, "plus", "minus")   # 8 spin x 3 = 24 start
STAGES = [{"wC_mul": 0.2, "lam_mul": 0.0, "iters": 25},
          {"wC_mul": 0.6, "lam_mul": 0.0, "iters": 25},
          {"wC_mul": 1.0, "lam_mul": 1.0, "iters": 60}]
VAL_EVERY = 5                        # 정렬된 pid 중 i%5==0 을 validation 으로


def log(*a):
    print(*a, flush=True)


# ================================================================ SSM (부분집합)
class SubsetSSM(object):
    def __init__(self, npz, use_pids=None):
        Z = np.load(npz, allow_pickle=True)
        pids = [str(p) for p in Z["pids"]]
        idx = list(range(len(pids))) if use_pids is None else \
            [i for i, p in enumerate(pids) if p in use_pids]
        self.X = Z["X"][idx]
        self.pids = [pids[i] for i in idx]
        self.gt_idx = int(Z["anchor_idx"][1])
        self.n, self.nv = self.X.shape[0], self.X.shape[1]
        F = self.X.reshape(self.n, -1)
        self.mu = F.mean(0)
        U, S, Vt = np.linalg.svd(F - self.mu, full_matrices=False)
        self.Vt = Vt
        self.sd = np.sqrt(np.maximum((S ** 2) / (self.n - 1), 0))
        self.ev = self.sd ** 2 / (self.sd ** 2).sum()

    def shape(self, c):
        v = self.mu.copy()
        if len(c):
            v = v + np.asarray(c) @ self.Vt[:len(c)]
        return v.reshape(self.nv, 3)

    def landmarks(self, c):
        P = self.shape(c)
        return {"femurHead": np.zeros(3), "greatTroch": P[self.gt_idx]}


# ================================================================ 합성 관측
def imaging_place(V, lm):
    """phase4b.imaging_frame 과 동일: ey=무릎->골두, ex=(head-gt)의 축수직성분,
       원점 = 메시 중심."""
    head, knee, gt = lm["femurHead"], lm["kneeCenter"], lm["greatTroch"]
    ey = head - knee; ey = ey / np.linalg.norm(ey)
    v = head - gt
    ex = v - np.dot(v, ey) * ey; ex = ex / np.linalg.norm(ex)
    R = np.vstack([ex, ey, np.cross(ex, ey)])
    c = V.mean(0)
    return (V - c) @ R.T, R, c


def rasterize(uv):
    col = np.round(uv[:, 0] / PIX + W / 2.0).astype(int)
    row = np.round(H / 2.0 - uv[:, 1] / PIX).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    m = np.zeros((H, W), bool)
    m[row[ok], col[ok]] = True
    m = ndi.binary_closing(m, np.ones((5, 5)))
    m = ndi.binary_fill_holes(m)
    lab, nl = ndi.label(m)
    if nl:
        sz = ndi.sum(m, lab, range(1, nl + 1))
        m = lab == (int(np.argmax(sz)) + 1)
    return m


def contour_from_mask(m, n=N_CONTOUR_PTS):
    cs = measure.find_contours(m.astype(float), 0.5)
    c = max(cs, key=len)
    uv = np.stack([(c[:, 1] - W / 2.0) * PIX, (H / 2.0 - c[:, 0]) * PIX], 1)
    p = np.vstack([uv, uv[:1]])
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def sdf_of(m):
    return (ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)) * PIX


class SynthObs(object):
    """validation subject 의 메시를 투영해 만든 관측. DRR 이 아니다."""

    def __init__(self, pid):
        V, _ = read_stl(os.path.join("ssm_raw", pid + ".stl"))
        V = dedup(V)
        lm = read_palp(os.path.join("ssm_raw", pid + "_palp.inp"))
        Vi, R, c = imaging_place(V, lm)
        self.pid, self.Vi = pid, Vi
        self.lm3d = {n: (lm[n] - c) @ R.T for n in ("femurHead", "greatTroch")}
        self.ang = [np.radians(a) for a in VIEWS]
        self.lm, self.contour, self.sdf = {}, [], []
        L = np.array([self.lm3d["femurHead"], self.lm3d["greatTroch"]])
        self.lm = {"femurHead": [], "greatTroch": []}
        for th in self.ang:
            q = project(L, th)
            self.lm["femurHead"].append(q[0]); self.lm["greatTroch"].append(q[1])
            m = rasterize(project(Vi, th))
            self.contour.append(contour_from_mask(m))
            self.sdf.append(sdf_of(m))
        self.lm = {n: np.array(v) for n, v in self.lm.items()}
        self.contour_fit = [c2[::max(1, len(c2) // N_OBS_CONTOUR)][:N_OBS_CONTOUR]
                            for c2 in self.contour]
        self.tree = cKDTree(Vi)

    def sample_sdf(self, i, uv):
        rc = np.stack([H / 2.0 - uv[:, 1] / PIX, uv[:, 0] / PIX + W / 2.0], 0)
        return ndi.map_coordinates(self.sdf[i], rc, order=1, mode="nearest")


# ================================================================ 역산
class ReconR(object):
    """ReconC + coefficient regularization.  E_reg = lambda * sum (c_i/sigma_i)^2"""

    LMN = ("femurHead", "greatTroch")

    def __init__(self, ssm, ob, k, wC, lam=0.0):
        self.S, self.ob, self.k, self.wC, self.lam = ssm, ob, k, wC, lam
        self.np_ = k + 7
        self.sub = np.arange(0, ssm.nv, max(1, ssm.nv // N_MESH_CONTOUR))[:N_MESH_CONTOUR]

    def _posed(self, p):
        c = p[7:] if self.k else []
        lm = self.S.landmarks(c)
        return (pose_apply(self.S.shape(c), p[:7]),
                pose_apply(np.array([lm[n] for n in self.LMN]), p[:7]))

    def residual(self, p, wC=None, lam=None):
        wC = self.wC if wC is None else wC
        lam = self.lam if lam is None else lam
        Pw, Lw = self._posed(p)
        Ps = Pw[self.sub]
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
        out = [rl / np.sqrt(len(rl))]
        if wC > 0:
            v = np.concatenate(rc)
            out.append(wC * v / np.sqrt(len(v)))
        if lam > 0 and self.k:
            out.append(np.sqrt(lam) * p[7:] / self.S.sd[:self.k])
        return np.concatenate(out)

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
        c0 = np.zeros(self.k)
        if cinit == "plus":
            c0 = 1.0 * self.S.sd[:self.k]
        elif cinit == "minus":
            c0 = -1.0 * self.S.sd[:self.k]
        return np.concatenate([[np.log(s)], rotvec_of(R), th - s * (R @ mh), c0])

    def _lm_step(self, p, wC, lam, iters):
        h = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]
                     + [max(1e-2, 1e-3 * self.S.sd[i]) for i in range(self.k)])
        r = self.residual(p, wC, lam); E = float(r @ r); mu = 1e-3; imp = 1e9
        for it in range(iters):
            J = np.zeros((len(r), self.np_))
            for i in range(self.np_):
                q = p.copy(); q[i] += h[i]
                J[:, i] = (self.residual(q, wC, lam) - r) / h[i]
            A = J.T @ J; g = -J.T @ r
            ok = False
            for _ in range(6):
                try:
                    d = np.linalg.solve(A + mu * np.diag(np.diag(A))
                                        + 1e-12 * np.eye(self.np_), g)
                except np.linalg.LinAlgError:
                    mu *= 10; continue
                pn = p + d
                rn = self.residual(pn, wC, lam); En = float(rn @ rn)
                if En < E:
                    imp = E - En; p, r, E = pn, rn, En
                    mu = max(mu * 0.35, 1e-10); ok = True; break
                mu *= 9
            if not ok or imp < 1e-12:
                break
        return p

    def fit(self, p0, strategy="A"):
        """A: 단일 단계 lambda=0 (6-B 재현)
           B: coarse-to-fine, lambda=0
           C: coarse-to-fine + lambda"""
        p = p0.copy()
        if strategy == "A":
            p = self._lm_step(p, self.wC, 0.0, 120)
        else:
            for st in STAGES:
                lam = self.lam * st["lam_mul"] if strategy == "C" else 0.0
                p = self._lm_step(p, self.wC * st["wC_mul"], lam, st["iters"])
        lam_fin = self.lam if strategy == "C" else 0.0
        r = self.residual(p, self.wC, lam_fin)
        # 보고용 목적함수는 항상 lambda 없이 잰다 (조건 간 비교 가능하도록)
        r0 = self.residual(p, self.wC, 0.0)
        return {"params": p, "obj_rms": float(np.sqrt((r0 ** 2).mean())),
                "obj_with_reg": float(np.sqrt((r ** 2).mean()))}

    def jac(self, p, lam=None):
        lam = self.lam if lam is None else lam
        h = np.array([1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05]
                     + [max(1e-2, 1e-3 * self.S.sd[i]) for i in range(self.k)])
        r0 = self.residual(p, self.wC, lam)
        J = np.zeros((len(r0), self.np_))
        for i in range(self.np_):
            q = p.copy(); q[i] += h[i]
            J[:, i] = (self.residual(q, self.wC, lam) - r0) / h[i]
        return J

    def zstats(self, p):
        if not self.k:
            return {"max": 0.0, "mean": 0.0, "n_gt2": 0, "n_gt3": 0}
        z = np.abs(p[7:] / self.S.sd[:self.k])
        return {"max": float(z.max()), "mean": float(z.mean()),
                "n_gt2": int((z > 2).sum()), "n_gt3": int((z > 3).sum())}


def surf_stats(P, tree):
    d = np.sort(tree.query(P)[0])
    return {"mean": round(float(d.mean()), 3), "median": round(float(np.median(d)), 3),
            "p95": round(float(np.percentile(d, 95)), 3), "max": round(float(d.max()), 3)}


def prox_mask(P):
    return P[:, 1] > (P[:, 1].max() - D1_EXTENT)


class Pat001Eval(object):
    """Pat001 GT 는 여기서만 열린다."""

    def __init__(self):
        E = json.load(io.open("ct_eval_mesh_Pat001.json", encoding="utf-8"))
        self.tree = cKDTree(np.array(E["verts"]))
        c = json.load(io.open("ct_case_Pat001_D1.json", encoding="utf-8"))
        self.gt3d = c["evaluation_only"]["landmarks_3d_imaging_mm"]

    def evaluate(self, ssm, p, k):
        P = pose_apply(ssm.shape(p[7:] if k else []), p[:7])
        lm = ssm.landmarks(p[7:] if k else [])
        L = pose_apply(np.array([lm["femurHead"], lm["greatTroch"]]), p[:7])
        return {"surface_whole": surf_stats(P, self.tree),
                "surface_proximal_d1": surf_stats(P[prox_mask(P)], self.tree),
                "landmark_3d_error_mm": {
                    "femurHead": round(float(np.linalg.norm(L[0] - self.gt3d["femurHead"])), 4),
                    "greatTroch": round(float(np.linalg.norm(L[1] - self.gt3d["greatTroch"])), 4)},
                "scale": round(float(np.exp(p[0])), 5)}
