# -*- coding: utf-8 -*-
"""
Phase 6-D / PART A : validation subject 에 Pat001 과 동일한 CT->DRR 파이프라인 적용.

Phase 6-C 의 validation 관측은 STL 정점 투영으로 합성한 것이라 Pat001 의 관측
(CT-derived DRR)과 채널이 달랐다. 여기서는 phase4b.py 와 완전히 동일한 코드로
validation subject 의 DRR 을 만든다.

  CT(.nrrd) -> 전문가 STL 복셀화 -> 마스크 밖 HU=-1000 -> mu -> 광선적분 DRR
  -> 0/30/60 deg -> 동일 검출기(450x600, 0.8mm) -> contour 추출 -> 2D 랜드마크 투영

phase4b.py 는 수정하지 않는다. 아래 함수들은 phase4b.py 에서 그대로 옮겨 적은 것이며
수식이 동일하다(voxelize / imaging_frame / rotY / proj_matrix / project_pts /
uv_to_px / render_drr / trilinear).

출력
  drr_val/PatNNN/drr_{000,030,060}deg.npy
  drr_val/PatNNN/case.json      reconstruction_input(2D) + evaluation_only(3D)
  drr_val/PatNNN/mesh_imaging.npy   평가용 메시 정점 (imaging frame)
"""
import sys, io, os, json, time, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import nrrd
from scipy import ndimage as ndi
from skimage import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RAW = "ssm_raw"
OUTD = "drr_val"
VAL = ["Pat002", "Pat019", "Pat031", "Pat042", "Pat060", "Pat080", "Pat095"]
assert "Pat001" not in VAL, "Pat001 은 validation 이 아니다"

# --- phase4b.py 와 동일한 촬영 규약 (그대로 옮김) --------------------------
SID, SOD = 1150.0, 1050.0
OID = SID - SOD
DET_W_MM, DET_H_MM = 360.0, 480.0
PIX = 0.8
VIEWS = [0.0, 30.0, 60.0]
MU_WATER = 0.0195
STEP = 0.4
THR = 0.02            # contour threshold, phase4e_contour 와 동일
N_CONTOUR = 240


def log(*a):
    print(*a, flush=True)


# ================= phase4b.py 에서 그대로 옮긴 함수들 =====================
def read_stl(path):
    b = io.open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    return a[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)


def voxelize(tri, shape, spacing, origin, samp=0.25):
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    nsub = np.maximum(2, np.ceil(np.sqrt(area) / samp).astype(int))
    shell = np.zeros(shape, bool)
    for k in np.unique(nsub):
        sel = tri[nsub == k]
        if len(sel) == 0:
            continue
        g = np.linspace(0.0, 1.0, k)
        uu, vv = np.meshgrid(g, g, indexing="ij")
        m = (uu + vv) <= 1.0
        uu, vv = uu[m], vv[m]
        a = sel[:, 0][:, None, :]
        d1 = (sel[:, 1] - sel[:, 0])[:, None, :]
        d2 = (sel[:, 2] - sel[:, 0])[:, None, :]
        pts = a + uu[None, :, None] * d1 + vv[None, :, None] * d2
        idx = np.round((pts.reshape(-1, 3) - origin) / spacing).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        idx = idx[ok]
        shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndi.label(~shell)
    outside = lab[0, 0, 0]
    inside = (~shell) & (lab != outside)
    return inside | shell, inside


def imaging_frame(lm, mesh_v):
    head = np.array(lm["femurHead"]); knee = np.array(lm["kneeCenter"])
    gt = np.array(lm["greatTroch"])
    ey = head - knee; ey = ey / np.linalg.norm(ey)
    v = head - gt
    ex = v - np.dot(v, ey) * ey; ex = ex / np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    R = np.vstack([ex, ey, ez])
    c = mesh_v.mean(0)
    return R, -R @ c


def rotY(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def proj_matrix(R, t, deg):
    Ry = rotY(deg)
    A = Ry @ R; b = Ry @ t
    P = np.zeros((3, 4))
    P[0, :3] = SID * A[0]; P[0, 3] = SID * b[0]
    P[1, :3] = SID * A[1]; P[1, 3] = SID * b[1]
    P[2, :3] = A[2]; P[2, 3] = b[2] + SOD
    return P


def project_pts(P, X):
    X = np.atleast_2d(X)
    h = np.hstack([X, np.ones((len(X), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


def trilinear(vol, f):
    sh = np.array(vol.shape)
    i0 = np.floor(f).astype(int)
    w = f - i0
    out = np.zeros(f.shape[:-1])
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = i0 + np.array([dz, dy, dx])
                ok = np.all((idx >= 0) & (idx < sh), axis=-1)
                wt = (np.where(dz, w[..., 0], 1 - w[..., 0]) *
                      np.where(dy, w[..., 1], 1 - w[..., 1]) *
                      np.where(dx, w[..., 2], 1 - w[..., 2]))
                ii = np.clip(idx, 0, sh - 1)
                out += np.where(ok, vol[ii[..., 0], ii[..., 1], ii[..., 2]] * wt, 0.0)
    return out


def render_drr(mu, spacing, origin, R, t, deg, aabb_img):
    W = int(round(DET_W_MM / PIX)); H = int(round(DET_H_MM / PIX))
    u = (np.arange(W) - W / 2.0 + 0.5) * PIX
    v = (H / 2.0 - np.arange(H) - 0.5) * PIX
    UU, VV = np.meshgrid(u, v)
    src = np.array([0.0, 0.0, -SOD])
    det = np.stack([UU, VV, np.full_like(UU, OID)], -1).reshape(-1, 3)
    dirs = det - src
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    lo, hi = aabb_img
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - src) / dirs
        t1 = (hi - src) / dirs
    tmin = np.nanmax(np.minimum(t0, t1), axis=1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=1)
    hit = tmax > np.maximum(tmin, 0)
    tmin = np.maximum(tmin, 0)
    nsteps = int(np.ceil((tmax[hit] - tmin[hit]).max() / STEP)) + 1
    acc = np.zeros(len(dirs))
    Rinv = R.T
    Ry_inv = rotY(-deg)
    idxs = np.where(hit)[0]
    CH = 4000
    for s in range(0, len(idxs), CH):
        sel = idxs[s:s + CH]
        d = dirs[sel]; a0 = tmin[sel]; a1 = tmax[sel]
        ts = a0[:, None] + (np.arange(nsteps)[None, :]) * STEP
        valid = ts <= a1[:, None]
        P_img = src[None, None, :] + ts[:, :, None] * d[:, None, :]
        P_un = P_img @ Ry_inv.T
        P_ct = (P_un - t) @ Rinv.T
        fidx = (P_ct - origin) / spacing
        acc[sel] = (trilinear(mu, fidx) * valid).sum(1) * STEP
    return acc.reshape(H, W)


# ================= contour 추출 (phase4e_contour.py 와 동일) ===============
def contour_of(img, n=N_CONTOUR):
    W = int(round(DET_W_MM / PIX)); H = int(round(DET_H_MM / PIX))
    m = img > THR
    m = ndi.binary_closing(m, np.ones((3, 3)))
    m = ndi.binary_fill_holes(m)
    lab, nl = ndi.label(m)
    if nl:
        sz = ndi.sum(m, lab, range(1, nl + 1))
        m = lab == (int(np.argmax(sz)) + 1)
    cs = measure.find_contours(m.astype(float), 0.5)
    c = max(cs, key=len)
    uv = np.stack([(c[:, 1] - W / 2.0) * PIX, (H / 2.0 - c[:, 0]) * PIX], 1)
    p = np.vstack([uv, uv[:1]])
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(tt, s, p[:, 0]), np.interp(tt, s, p[:, 1])], 1), m


def read_palp(path):
    d = {}
    for line in io.open(path, encoding="utf-8", errors="replace"):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = float(v.strip())
    return {n: np.array([d[n + "X"], d[n + "Y"], d[n + "Z"]])
            for n in ("femurHead", "headDir", "greatTroch", "kneeCenter")}


# ================================================================ 실행
if __name__ == "__main__":
    if not os.path.isdir(OUTD):
        os.makedirs(OUTD)
    T00 = time.time()
    for si, pid in enumerate(VAL):
        od = os.path.join(OUTD, pid)
        if not os.path.isdir(od):
            os.makedirs(od)
        if os.path.exists(os.path.join(od, "case.json")):
            log("[%d/%d] %s 이미 완료" % (si + 1, len(VAL), pid))
            continue
        t0 = time.time()
        vol, hdr = nrrd.read(os.path.join(RAW, pid + ".nrrd"))
        sd = np.array(hdr["space directions"], float)
        spacing = np.abs(np.diag(sd)); origin = np.array(hdr["space origin"], float)
        tri = read_stl(os.path.join(RAW, pid + ".stl"))
        mesh_v = tri.reshape(-1, 3)
        mask, inner = voxelize(tri, vol.shape, spacing, origin)
        hu = np.where(mask, vol, -1000).astype(np.float32)
        mu = np.clip(MU_WATER * (1.0 + hu / 1000.0), 0, None).astype(np.float32)
        lm = read_palp(os.path.join(RAW, pid + "_palp.inp"))
        R, t = imaging_frame(lm, mesh_v)
        Vimg = mesh_v @ R.T + t
        np.save(os.path.join(od, "mesh_imaging.npy"),
                np.unique(np.round(Vimg, 4), axis=0).astype(np.float32))
        views, lm2d = [], {}
        for deg in VIEWS:
            Vrot = Vimg @ rotY(deg).T
            aabb = (Vrot.min(0) - 2.0, Vrot.max(0) + 2.0)
            img = render_drr(mu, spacing, origin, R, t, deg, aabb)
            tag = "%03ddeg" % int(deg)
            np.save(os.path.join(od, "drr_%s.npy" % tag), img.astype(np.float32))
            cont, m = contour_of(img)
            P = proj_matrix(R, t, deg)
            uv = project_pts(P, np.array([lm["femurHead"], lm["greatTroch"]]))
            views.append({"view_angle_deg": float(deg),
                          "landmarks_2d_mm": {"femurHead": uv[0].tolist(),
                                              "greatTroch": uv[1].tolist()},
                          "contour_uv_mm": np.round(cont, 4).tolist(),
                          "mask_area_px": int(m.sum()),
                          "drr_max": float(img.max()),
                          "drr_zero_fraction": float((img == 0).mean())})
        json.dump({"subject": pid,
                   "modality": "CT-derived synthetic radiograph (DRR). NOT a real X-ray.",
                   "pipeline": "phase4b.py 와 동일한 코드/수식",
                   "reconstruction_input": {
                       "landmark_ids": ["femurHead", "greatTroch"],
                       "known_view_angles_deg": VIEWS,
                       "SID_mm": SID, "SOD_mm": SOD,
                       "detector": {"width_px": int(round(DET_W_MM / PIX)),
                                    "height_px": int(round(DET_H_MM / PIX)),
                                    "pixel_spacing_mm": PIX},
                       "contour_threshold": THR,
                       "views": views},
                   "evaluation_only": {
                       "landmarks_3d_imaging_mm": {
                           n: ((lm[n] @ R.T) + t).tolist()
                           for n in ("femurHead", "greatTroch")},
                       "mesh_file": "mesh_imaging.npy",
                       "note": "이 블록은 평가에서만 쓴다."}},
                  io.open(os.path.join(od, "case.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        log("[%d/%d] %s  CT %s  복셀화 %.1f cm3  %.0f s  (누적 %.0f s)"
            % (si + 1, len(VAL), pid, vol.shape,
               mask.sum() * float(np.prod(spacing)) / 1000,
               time.time() - t0, time.time() - T00))
    log("완료 %.0f 초" % (time.time() - T00))
