# Femur-3D Research Demo (web)

STEP38에서 최종 lock 된 모델(Main = M1, Fallback = F1)의 연구 결과를 보여주는 웹 데모입니다.
**연구 코드(STEP*/phase*/drr_g/ssm_raw)는 읽기 전용으로만 사용하며, 이 디렉터리와 `femur-ct/demo_data/` 밖으로는 아무것도 쓰지 않습니다.**

```
femur-ct/
├── web/
│   ├── export_demo_data.py     # 연구 산출물 → demo_data (읽기 전용 변환)
│   ├── frontend/               # React + TypeScript + Vite + Three.js(R3F)
│   ├── backend/                # FastAPI — 잠금 파이프라인 live inference
│   └── _shots/                 # headless Chrome 검증 스크린샷
└── demo_data/                  # 생성물: meshes/, drr/, research.json, manifest.json
```

## 1. demo data 생성 (최초 1회)

```bash
cd femur-ct
python web/export_demo_data.py
```

- validation 6명(Pat019/031/042/060/080/095)의 **DRR 3장 → PNG**, **복원 메시 3종 + GT 메시 → STL**,
  SNR50 비교용 메시 2종, 그리고 모든 지표를 `demo_data/` 에 씁니다.
- `demo_data/` 는 `frontend/public/demo-data/` 로 복사되어 개발/빌드 양쪽에서 서빙됩니다.
- 정합 규약: 복원 메시는 `pose_apply(Xn_hat, E0 pose)`, GT는 `imaging_frame(STL)` — **평가에 쓰인 것과 같은 변환**이고,
  두 메시에 같은 display 정규화(GT bbox 중심, 최대 반경 = 1)를 적용합니다.

## 2. 프론트엔드

```bash
cd web/frontend
npm install
npm run dev        # http://localhost:5180
npm run build      # dist/
npm run preview    # http://localhost:5181
```

## 3. 백엔드 (선택 — live inference)

```bash
cd web/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8018
```

백엔드가 떠 있으면 프론트의 Reconstruction 페이지가 `LIVE BACKEND` 로 바뀌고,
`RUN RECONSTRUCTION` 이 서버에서 **실제 잠금 파이프라인**(M1 adaptive threshold → contour/SDF → 잠금 B0 추론)을
돌려 새 메시를 생성해 뷰어에 표시합니다. 백엔드가 없으면 자동으로 `DEMO MODE`(미리 계산된 결과)로 동작합니다.

### API

| method | path | 설명 |
|---|---|---|
| GET | `/api/health` | `live_inference` 가능 여부, 모델 md5 |
| GET | `/api/cases` | demo_data 케이스 목록·지표 |
| POST | `/api/reconstruct` | `{case_id, condition: clean\|missing\|fallback}` → 서버 재추론 결과 (mesh_url, latent, threshold 로그, 타이밍) |
| POST | `/api/reconstruct/upload` | DRR `.npy` 3장(600×450) 업로드 → 같은 파이프라인 |
| GET | `/api/mesh/{token}` | 생성된 STL |

검증: 저장된 DRR로 서버가 만든 메시는 연구 결과(STEP37 predictions)와 **최대 차 0.000000 mm** 로 일치합니다.

## 4. 주의

- 본 사이트는 연구용 시제품이며 의료 진단 시스템이 아닙니다.
- 모든 입력은 CT에서 생성한 DRR이고, 실제 X-ray로는 검증되지 않았습니다.
- 2D landmark는 영상 검출값이 아니라 3D 좌표의 투영값입니다(연구 규약 그대로).
