# -*- coding: utf-8 -*-
"""
Phase 6-G 공통 모듈 : 임의 각도 조합 관측, 임의 mode 부분집합 shape 역산, observability.

기존 Phase 6-A~F 파일은 수정하지 않는다.
ReconE(pose-only) / ReconF(shape-only) 는 그대로 import 해서 쓴다.
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import scipy.ndimage as ndi
from phase6.phase6_ssm_xray import pose_apply, project
from phase6.phase6f_shape_only import ReconF
from phase6.phase6c_reg import surf_stats, prox_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P6 = "phase6"
PIX, W, H = 0.8, 450, 600
N_OBS_CONTOUR = 120
SEED = 20260911

VIEW_COUNTS = {2: [0.0, 60.0], 3: [0.0, 30.0, 60.0], 4: [0.0, 20.0, 40.0, 60.0],
               5: [0.0, 15.0, 30.0, 45.0, 60.0],
               7: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}
SPANS = {10: [0.0, 5.0, 10.0], 20: [0.0, 10.0, 20.0], 30: [0.0, 15.0, 30.0],
         45: [0.0, 22.5, 45.0], 60: [0.0, 30.0, 60.0], 90: [0.0, 45.0, 90.0]}
NOISES = [0.0, 0.5, 1.0, 2.0]


def log(*a):
    print(*a, flush=True)


class GObs(object):
    """drr_g 의 임의 각도 부분집합을 관측으로 만든다. 잡음은 결정적 seed."""

    def __init__(self, pid, angles, noise=0.0, seed=SEED):
        d = os.path.join("drr_g", pid)
        C = json.load(io.open(os.path.join(d, "case.json"), encoding="utf-8"))
        RI = C["reconstruction_input"]
        by = {round(v["view_angle_deg"], 3): v for v in RI["views"]}
        self.pid, self.angles_deg = pid, list(angles)
        vs = [by[round(a, 3)] for a in angles]
        rng = np.random.default_rng(seed + int(1000 * sum(angles)) + int(100 * noise))
        self.ang = [np.radians(a) for a in angles]
        lm = {"femurHead": [], "greatTroch": []}
        self.contour, self.sdf = [], []
        for v in vs:
            for n in lm:
                p = np.array(v["landmarks_2d_mm"][n], float)
                if noise > 0:
                    p = p + rng.normal(0, noise, 2)
                lm[n].append(p)
            c = np.array(v["contour_uv_mm"], float)
            if noise > 0:
                c = c + rng.normal(0, noise, c.shape)
            self.contour.append(c)
            a = np.load(os.path.join(d, "drr_%s.npy" % v["tag"])).astype(float)
            m = a > RI["contour_threshold"]
            m = ndi.binary_closing(m, np.ones((3, 3)))
            m = ndi.binary_fill_holes(m)
            lab, nl = ndi.label(m)
            if nl:
                sz = ndi.sum(m, lab, range(1, nl + 1))
                m = lab == (int(np.argmax(sz)) + 1)
            if noise > 0:                      # 마스크도 같은 규모로 흐리게
                sdf = (ndi.distance_transform_edt(~m)
                       - ndi.distance_transform_edt(m)) * PIX
                sdf = sdf + rng.normal(0, noise, sdf.shape) * 0.0   # 경계 잡음은 contour 로만
            else:
                sdf = (ndi.distance_transform_edt(~m)
                       - ndi.distance_transform_edt(m)) * PIX
            self.sdf.append(sdf)
        self.lm = {n: np.array(v) for n, v in lm.items()}
        self.contour_fit = [c[::max(1, len(c) // N_OBS_CONTOUR)][:N_OBS_CONTOUR]
                            for c in self.contour]
        self.noise = noise
        self.__mesh = np.load(os.path.join(d, "mesh_imaging.npy")).astype(float)
        self.__gt3d = C["evaluation_only"]["landmarks_3d_imaging_mm"]

    def sample_sdf(self, i, uv):
        rc = np.stack([H / 2.0 - uv[:, 1] / PIX, uv[:, 0] / PIX + W / 2.0], 0)
        return ndi.map_coordinates(self.sdf[i], rc, order=1, mode="nearest")

    # --- 평가 전용 ---
    def eval_mesh(self):
        return self.__mesh

    def eval_gt3d(self):
        return self.__gt3d


class ReconG(ReconF):
    """ReconF 와 동일하되 임의의 mode 인덱스 부분집합을 쓴다."""

    def __init__(self, ssm, ob, pose, mode_idx, lam=0.0, wC=0.25):
        idx = np.asarray(mode_idx, int)
        ReconF.__init__(self, ssm, ob, pose, len(idx), lam, wC)
        self.mode_idx = idx
        self.B = ssm.Vt[idx]
        self.sd = ssm.sd[idx]


# ================================================================ G-1 지표
def rasterize_mask(uv):
    col = np.round(uv[:, 0] / PIX + W / 2.0).astype(int)
    row = np.round(H / 2.0 - uv[:, 1] / PIX).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    m = np.zeros((H, W), bool)
    m[row[ok], col[ok]] = True
    m = ndi.binary_closing(m, np.ones((5, 5)))
    return ndi.binary_fill_holes(m)


def edge_points(uv, nb=70):
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


def mode_observability(ssm, pose, angles, pc, sub_idx):
    """PC 하나를 +-1 sigma 움직였을 때 각 view 에서 관측이 얼마나 변하는가.
       GT 를 쓰지 않는다 (모델과 촬영 기하만)."""
    mu = ssm.mu
    out = []
    for deg in angles:
        th = np.radians(deg)
        P0 = pose_apply((mu).reshape(ssm.nv, 3)[sub_idx], pose)
        uv0 = project(P0, th)
        m0 = rasterize_mask(uv0)
        e0 = edge_points(uv0)
        L0 = project(pose_apply(np.array([np.zeros(3),
                                          mu.reshape(ssm.nv, 3)[ssm.gt_idx]]), pose), th)
        d_edge, d_max, d_area, d_iou, d_lm = [], [], [], [], []
        for sgn in (1.0, -1.0):
            a = np.zeros(len(mu))
            v = (mu + sgn * ssm.sd[pc] * ssm.Vt[pc]).reshape(ssm.nv, 3)
            P1 = pose_apply(v[sub_idx], pose)
            uv1 = project(P1, th)
            m1 = rasterize_mask(uv1)
            e1 = edge_points(uv1)
            dd = np.sqrt(((e1[:, None, :] - e0[None, :, :]) ** 2).sum(-1)).min(1)
            d_edge.append(float(dd.mean())); d_max.append(float(dd.max()))
            a0, a1 = m0.sum(), m1.sum()
            d_area.append(float(100.0 * abs(a1 - a0) / max(a0, 1)))
            inter = np.logical_and(m0, m1).sum(); uni = np.logical_or(m0, m1).sum()
            d_iou.append(float(1.0 - inter / max(uni, 1)))
            L1 = project(pose_apply(np.array([np.zeros(3), v[ssm.gt_idx]]), pose), th)
            d_lm.append(float(np.linalg.norm(L1 - L0, axis=1).mean()))
        out.append({"view_deg": float(deg),
                    "edge_mean_mm": round(float(np.mean(d_edge)), 4),
                    "edge_max_mm": round(float(np.mean(d_max)), 4),
                    "area_change_pct": round(float(np.mean(d_area)), 4),
                    "iou_loss": round(float(np.mean(d_iou)), 5),
                    "landmark_shift_mm": round(float(np.mean(d_lm)), 4)})
    return out
