# DEPLOYMENT_FILE_MANIFEST — 배포 repo 파일 목록 (femur-ct-deploy/)

원본 `femur-ct/`는 그대로 보존. 이 repo는 **복사본**이며, 파일 이동·삭제는 일절 없다.
선정 기준: backend `run_pipeline` 실행 시점의 실제 import + 파일 읽기 추적, + 배포 복사본 실기동 검증.
검증: 배포 복사본에서 Pat019 clean / Pat080 fallback / Pat060 missing 재추론 →
연구 precomputed 대비 max 차이 **0.000000 mm** (3/3).

## A. 반드시 포함 — backend 실행 코드

| 경로 | 이유 |
|---|---|
| `web/backend/app/main.py` | FastAPI 엔드포인트 + 파이프라인 호출부 |
| `web/backend/requirements.txt` | fastapi/uvicorn/numpy/scipy/scikit-image/pillow/python-multipart/**pynrrd** (pynrrd는 원본에 빠져 있어 배포용으로 추가 — 없으면 live 503) |
| `web/backend/.python-version` | Render Python 3.13 고정 |
| `STEP35/code/s35_common.py` | M1 adaptive threshold + contour (import 시점) |
| `STEP34/code/s34_common.py` | s35가 import. **import 시점에 `STEP33/results/intensity_stats.json`을 읽는다** |
| `STEP33/code/s33_common.py` | SDF/LMK 블록 + QC (`blocks_and_qc`, 매 reconstruct 호출) |
| `STEP26/code/s26_model.py` | 잠금 B0 추론 (K=15, γ=0.001) |
| `STEP37/code/s37_common.py` | F1 추론식 (K=20, γ=0.01, fallback 시 지연 import) |
| `STEP21/code/s21_common.py` | s33.D17이 import (descriptor 소스 추출용) |
| `STEP23/code/s23_common.py` | `blocks_and_qc` 내부 import (SDF grid) |
| `STEP17/code/s17_03_descriptors.py` | **소스 그대로 읽어서** frame/norm/knee_2d 추출 (C21.extract) |
| `phase5_correspond.py` | `read_palp` (palp 파일 읽기) |
| `phase6/phase6_ssm_xray.py` | `pose_apply`/`project` (mesh 좌표 변환) |
| `phase6/phase6d_validation_drr.py` | `imaging_frame`/`read_stl` + 상수 (import 시 `nrrd` 필요) |
| `phase6/phase6h_extend.py` | `_imaging_of` (매 reconstruct마다 `ssm_raw/<pid>.stl` 읽기) |
| `phase6/phase6c_reg.py`, `phase6g_obs.py`, `phase6b_ssm_contour.py`, `phase6f_shape_only.py` | phase6h import 연쇄 (모듈 import 시점에 필요) |

## B. 배포 시 필요한 데이터 (런타임 읽기)

| 경로 | 이유 | 용량 |
|---|---|---|
| `drr_g/Pat{019,031,042,060,080,095}/{case.json,drr_000.0deg.npy,drr_045.0deg.npy,drr_090.0deg.npy}` | VIEW_TAGS 3장 + landmark 투영값 (케이스당 4파일만) | ~20MB |
| `ssm_raw/Pat*.stl` (6개) | `_imaging_of`가 imaging frame 계산용으로 매 추론마다 읽음 | ~36MB |
| `ssm_raw/*_palp.inp` (6개) | `read_palp` (3D 기준점) | 수 KB |
| `STEP26/final_model/b0_final_model.npz` | 잠금 B0 (md5 검증 포함) | 6.4MB |
| `STEP33/results/intensity_stats.json` | s34 import 시점 읽기 (없으면 engine 503 — 실측 확인됨) | 21KB |
| `STEP12/results/_ck.json` | E0 pose | 406KB |
| `phase6/_6j_faces_ssm.npy` | SSM triangle topology | 116KB |
| `demo_data/` 전체 | backend는 `manifest.json`만 읽음. meshes/PNG는 backend 미사용이나 manifest 참조 무결성상 통째로 포함 | 17MB |
| `web/frontend/` (node_modules/dist 제외) | Pages 빌드용 소스 + `public/demo-data` (DEMO MODE) | ~2MB |

## C. 제외 가능한 대용량 파일

| 제외 | 이유 |
|---|---|
| `STEP17/data/ct_train/*.nrrd` (~1.4GB), `ssm_raw/*.nrrd` | 학습용 CT. reconstruct 실행 경로에서 읽지 않음 |
| `drr_g` 000/045/090 외 각도, `drr_val/`, `ground_truth/` | 파이프라인이 읽지 않음 |
| `STEP23/stageC`, `STEP13/14/18/32/33/34` 결과물 일체 | 실험 기록. intensity_stats.json만 예외 포함 |
| `ssm_raw` GT STL 중 6명 외 | `_imaging_of`는 요청 pid만 읽음 |
| `web/frontend/dist/`, `node_modules/` | Pages가 빌드. `web/backend/_runs/`, 로그, `__pycache__` |

## D. 제외 이유
런타임 import·파일 읽기 추적에 걸리지 않은 것은 전부 제외. 추측 제외 없음 —
`intensity_stats.json`은 첫 배포 복사본 기동 실패(503)로 **실측 후 추가**했다.

## E. 예상 repository size
- **약 95MB, 184 파일. 최대 단일 파일 7.5MB** (ssm_raw Pat095.stl).
- GitHub 제한(경고 50MB / 차단 100MB)에 걸리는 파일 없음. **Git LFS 불필요.**
- Render 런타임 추가분: pip 패키지(numpy/scipy/scikit-image/fastapi/pillow/multipart/pynrrd/uvicorn) 설치 시 수백 MB 디스크 사용. Render Web Service 빌드/실행 범위 내이며 Docker 불필요.
