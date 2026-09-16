# Femur-3D Research Demo — 배포용 repository

3-view DRR → 3D femur 복원 연구(M1/B0 + F1 fallback)의 공개 배포본.
원본 연구(`femur-ct/`)는 건드리지 않고, 실행에 필요한 파일만 복사했다.
자세한 선정 근거: `DEPLOYMENT_FILE_MANIFEST.md`.

## 목적
- 대학 의료영상/컴퓨터비전 연구실의 연구 시각화 도구 (Research Prototype)
- 의료 진단 시스템이 아니다. 임상 주장을 하지 않는다.

## 구성
- Cloudflare Pages: `web/frontend` (React 19 + Vite + Three.js) + DEMO MODE 데이터
- Render Web Service: `web/backend` (FastAPI, 잠금 M1 파이프라인 live 추론)

## 실행 방법 (로컬)
```bash
# backend (이 디렉터리에서)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8018 --app-dir web/backend
# frontend
cd web/frontend && npm install && npm run dev   # http://localhost:5180
```

## 환경변수
- Frontend: `VITE_API_BASE_URL` (배포 시 Render URL. dev는 비워 두면 same-origin proxy)
- Backend: `ALLOWED_ORIGINS` (배포 시 Pages 도메인. 미설정 시 `*`), `PORT` (Render 자동)

## Render 배포값
- Root Directory: repo root (이 디렉터리)
- Build: `pip install -r web/backend/requirements.txt`
- Start: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir web/backend`
- Health Check: `/api/health` (live_inference:true, md5 `83bbde9b…` 확인)

## Cloudflare Pages 배포값
- Root: `web/frontend`, Build: `npm run build`, Output: `dist`
- Env: `VITE_API_BASE_URL=https://<render-app>.onrender.com`

## 연구 모델의 한계
- 독립 검증 6명, CT→DRR 기반. 실제 X-ray 미검증.
- F1은 GT+KC 동시 missing 전용 제한적 fallback (p95 악화 잔존).
- 상세: 원본 `STEP38/FINAL_REPORT.md`, `MODEL_CARD.md` 참조.
