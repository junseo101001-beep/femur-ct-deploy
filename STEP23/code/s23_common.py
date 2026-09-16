# -*- coding: utf-8 -*-
"""STEP 23 공통 : 데이터 로드 (N27 만), 2D SDF 표현, 블록 정규화·joint PCA, 추정기, 표면 평가."""
import sys, io, os, json
_H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_H))
for p in (ROOT, os.path.join(ROOT, "STEP21", "code")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)
import numpy as np
import s21_common as C21                     # 감사 hook, STEP17 함수 소스 추출 (읽기 전용 import)

S23 = "STEP23"
HELD = C21.HELD
TAGS = ["000.0deg", "045.0deg", "090.0deg"]
U_GRID = np.linspace(-0.25, 0.25, 40)
V_GRID = np.linspace(-0.7, 0.7, 112)
LAMS = 10.0 ** np.linspace(-3, 3, 13)
KS = [3, 4, 5, 6, 8, 10]


def load():
    T = np.load(os.path.join("STEP17", "training", "targets_train.npz"))
    D = np.load(os.path.join("STEP17", "descriptors", "descriptors_train.npz"))
    pids = [str(p) for p in T["pids"]]
    assert pids == [str(p) for p in D["pids"]] and len(pids) == 27 and not set(pids) & set(HELD)
    return {"pids": pids, "Xn": T["Xn"].reshape(27, -1, 3).astype(float), "Y": T["alpha_sigma"], "mu": T["mu"], "V6": T["V6"], "sd": T["sd"],
            "D2": D["D2"], "LMK": D["D2"][:, 420:]}


# ---------------------------------------------------------------- 2D SDF
def sdf_grid(Q):
    """정규화 윤곽 다각형 Q (n×2, 닫힘) 의 부호 거리장 (안쪽 음수) 을 V_GRID×U_GRID 격자에서."""
    from skimage.measure import points_in_poly
    uu, vv = np.meshgrid(U_GRID, V_GRID)
    G = np.stack([uu.ravel(), vv.ravel()], 1)
    A = Q; B = np.roll(Q, -1, axis=0); AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    t = np.clip(((G[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    P = A[None] + t[..., None] * AB[None]
    d = np.sqrt(((G[:, None, :] - P) ** 2).sum(2)).min(1)
    inside = points_in_poly(G, Q)
    return np.where(inside, -d, d).reshape(len(V_GRID), len(U_GRID))


def sdf_subject(pid, D17):
    case = json.load(io.open(os.path.join("STEP17", "data", "drr_train", pid, "case.json"), encoding="utf-8"))
    views = {v["tag"]: v for v in case["reconstruction_input"]["views"]}
    out, chk = [], []
    for t in TAGS:
        C = np.array(views[t]["contour_uv_mm"], float)
        head = np.array(views[t]["landmarks_2d_mm"]["femurHead"])
        c, R, L = D17["frame"](C, head)
        Q = D17["norm"](C, c, R, L)
        out.append(sdf_grid(Q))
        chk.append({"frac_contour_outside_grid": float(((Q[:, 0] < U_GRID[0]) | (Q[:, 0] > U_GRID[-1]) | (Q[:, 1] < V_GRID[0]) | (Q[:, 1] > V_GRID[-1])).mean()),
                    "u_range": [float(Q[:, 0].min()), float(Q[:, 0].max())], "v_range": [float(Q[:, 1].min()), float(Q[:, 1].max())]})
    return np.array(out), chk


# ---------------------------------------------------------------- joint model
def blocks_of(X3, S, LMK=None):
    """X3 (n,14733), S (n,3,112,40), LMK (n,18) → 블록 dict (원 값)."""
    b = {"3D": X3.reshape(len(X3), -1)}
    for v, t in enumerate(TAGS):
        b["SDF_" + t] = S[:, v].reshape(len(S), -1)
    if LMK is not None:
        b["LMK"] = LMK
    return b


class Joint:
    """블록별 feature z-score / √d (fold training) → 연결 → SVD."""

    def __init__(self, blocks):
        self.names = list(blocks)
        self.stats = {}
        parts = []
        for k in self.names:
            B = blocks[k]
            m = B.mean(0); s = B.std(0); s = np.where(s < 1e-8, 1.0, s)
            d = B.shape[1]
            self.stats[k] = (m, s, np.sqrt(d))
            parts.append((B - m) / s / np.sqrt(d))
        Z = np.hstack(parts)
        self.n = len(Z)
        U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        self.Vt, self.sv = Vt, sv
        self.scores = U * sv
        self.var = sv ** 2 / (self.n - 1)
        self.slices, o = {}, 0
        for k in self.names:
            d = blocks[k].shape[1]; self.slices[k] = slice(o, o + d); o += d
        self.Z = Z

    def norm_block(self, k, B):
        m, s, r = self.stats[k]
        return (B - m) / s / r

    def denorm_3d(self, Z3):
        m, s, r = self.stats["3D"]
        return m + Z3 * s * r

    def lift_3d(self, z, K):
        return self.denorm_3d(np.atleast_2d(z)[:, :K] @ self.Vt[:K, self.slices["3D"]])

    def block_ev(self, K=10):
        """성분 k 가 블록 b 분산 중 설명하는 몫."""
        out = {}
        for k in self.names:
            sl = self.slices[k]
            tot = (self.Z[:, sl] ** 2).sum()
            out[k] = [float((self.sv[j] ** 2) * (self.Vt[j, sl] ** 2).sum() / tot) for j in range(min(K, len(self.sv)))]
        return out

    def names_2d(self):
        return [k for k in self.names if k != "3D"]

    def x2d(self, blocks):
        return np.hstack([self.norm_block(k, blocks[k]) for k in self.names_2d()])

    def W2d(self, K):
        return np.hstack([self.Vt[:K, self.slices[k]] for k in self.names_2d()])

    def proj(self, X2, K, gamma):
        W = self.W2d(K)
        A = W @ W.T + gamma * np.diag(1.0 / np.maximum(self.var[:K], 1e-12))
        return np.linalg.solve(A, W @ np.atleast_2d(X2).T).T


# ---------------------------------------------------------------- ridge (s17_04 과 같은 식, λ 마다 gram 재사용)
def ridge_fit_multi(X, Y, lams):
    mx, sx = X.mean(0), X.std(0); sx = np.where(sx < 1e-12, 1.0, sx)
    Xs = (X - mx) / sx; my = Y.mean(0); G = Xs @ Xs.T
    out = []
    for lam in lams:
        A = np.linalg.solve(G + lam * np.eye(len(X)), Y - my)
        out.append({"mx": mx, "sx": sx, "my": my, "W": Xs.T @ A, "lam": float(lam)})
    return out


def ridge_predict(m, X):
    return ((np.atleast_2d(X) - m["mx"]) / m["sx"]) @ m["W"] + m["my"]


# ---------------------------------------------------------------- 평가
def mean_vertex_dist(A, B):
    return float(np.linalg.norm(A.reshape(-1, 3) - B.reshape(-1, 3), axis=1).mean())


def z_eq(Xhat, data):
    return ((Xhat.reshape(len(Xhat), -1) - data["mu"]) @ data["V6"].T) / data["sd"]


class SurfaceEval:
    def __init__(self, pids):
        from phase6.phase6j_render import build_faces
        from phase6.phase6d_validation_drr import imaging_frame, read_stl, read_palp
        from phase5_correspond import frame_of, read_palp as rp5
        from scipy.spatial import cKDTree
        self.F = build_faces()
        self.Lref = json.load(io.open(os.path.join("STEP20", "landmarks", "landmark_checks.json"), encoding="utf-8"))["Lref_median_FL_mm"]
        self.sub = {}
        for p in pids:
            Ra, Ha, FL = frame_of(rp5(os.path.join("ssm_raw", p + "_palp.inp")))
            tri = read_stl(os.path.join("ssm_raw", p + ".stl"))
            Ri, ti = imaging_frame(read_palp(os.path.join("ssm_raw", p + "_palp.inp")), tri.reshape(-1, 3))
            gt = np.load(os.path.join("STEP17", "data", "drr_train", p, "mesh_imaging.npy")).astype(float)
            vol = abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)) / 1000.0
            self.sub[p] = {"Ra": Ra, "Ha": Ha, "FL": FL, "Ri": Ri, "ti": ti, "gt": gt, "tree": cKDTree(gt), "vol": vol}

    def metrics(self, pid, Xn_hat):
        from scipy.spatial import cKDTree
        from phase6.phase6j_render import mesh_volume
        s = self.sub[pid]
        P = ((Xn_hat.reshape(-1, 3) * (s["FL"] / self.Lref)) @ s["Ra"].T + s["Ha"]) @ s["Ri"].T + s["ti"]
        d1 = s["tree"].query(P)[0]; d2 = cKDTree(P).query(s["gt"])[0]
        v = abs(mesh_volume(P, self.F)) / 1000.0
        return {"sym": 0.5 * float(np.median(d1) + np.median(d2)), "sym_mean": 0.5 * float(d1.mean() + d2.mean()),
                "p95": float(max(np.percentile(d1, 95), np.percentile(d2, 95))), "max": float(max(d1.max(), d2.max())),
                "cov5": float(100 * (d2 < 5).mean()), "vol_err_pct": float(100 * (v - s["vol"]) / s["vol"])}


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, io.open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
