/**
 * 모든 성능 수치는 빌드 시점에 하드코딩하지 않고, 연구 산출물에서 생성된
 * /demo-data/research.json · manifest.json 을 런타임에 읽어서 사용한다.
 * (생성기: web/export_demo_data.py — STEP26/36/37/38/39 결과 파일이 source of truth)
 *
 * 이 파일에는 "설명 텍스트" 와 "구조 정의" 만 둔다. 숫자는 두지 않는다.
 */

export interface Metric { sym: number; p95: number; cov5: number; vol_err_pct?: number; alpha_rmse_sigma?: number; rot_deg?: number; trans_mm?: number }
export interface MeshInfo { file: string; triangles: number; bytes?: number; note?: string }
export interface DrrInfo { file: string; angle_deg: number; shape: number[]; raw_min: number; raw_max: number }

export interface CaseData {
  id: string
  pid: string
  label: string
  meshes: Record<string, MeshInfo>
  drr: Record<string, DrrInfo>
  metrics: {
    clean_m1: Metric
    missing_main: Metric
    fallback_f1: Metric
    baseline_e0: Metric
    delta_f1_minus_control: number
    delta_b0_minus_e0: number
  }
  display_transform: { center_mm: number[]; scale_mm: number }
  pose_source: string
}

/**
 * 검증 케이스 표시명. UI 표시 전용 — 데이터·순서·대응관계는 바꾸지 않는다.
 * 번호는 manifest.cases / validation_cohort 배열 순서(Pat019→1 … Pat095→6)와 동일하다.
 */
export function caseLabel(pid: string, idx: number): string {
  return `검증 대상 ${idx + 1} · ${pid}`
}

export interface Manifest {
  generated_at: string
  cases: CaseData[]
  noise_demo: null | {
    case_pid: string; snr: number; sigma: number
    meshes: Record<string, MeshInfo>; drr: Record<string, DrrInfo>; note: string
  }
  mesh_convention: { units: string; alignment: string; recon_topology: string }
}

export interface Research {
  generated_at: string
  source_of_truth: string[]
  lock: any
  final_rows: { category: string; condition: string; model: string; sym: number | null; p95: number | null; cov5: number | null; abs_vol: number | null; source: string }[]
  validation_cohort: string[]
  clean: {
    e0: { sym: number; p95: number; cov5: number; abs_vol: number; alpha: number }
    b0: { sym: number; p95: number; cov5: number; abs_vol: number; alpha: number }
    delta_b0_e0: number
    n_better: number
    per_subject: Record<string, { e0: number; b0: number; delta: number; improved: boolean; p95_b0: number; p95_e0: number; cov5_b0: number; vol_b0: number }>
  }
  missing: {
    control: any; f1: any; clean_m1: any
    delta_f1_control: { mean: number; median: number; sd: number; per_subject: Record<string, number>; max_single_gain_share: number | null }
    criteria: Record<string, any>
    verdict: string
    per_subject: Record<string, { clean: Metric; control: Metric; f1: Metric; delta: number }>
    geom_probe_step39: { sym: number | null; f0_sym?: number; verdict: string; adopted: boolean; adopted_fallback: string }
  }
  noise_demo: Manifest['noise_demo']
}

export const DEMO_BASE = '/demo-data'

export async function loadResearch(): Promise<{ research: Research; manifest: Manifest }> {
  const [r, m] = await Promise.all([
    fetch(`${DEMO_BASE}/research.json`).then((x) => { if (!x.ok) throw new Error(`research.json ${x.status}`); return x.json() }),
    fetch(`${DEMO_BASE}/manifest.json`).then((x) => { if (!x.ok) throw new Error(`manifest.json ${x.status}`); return x.json() }),
  ])
  return { research: r as Research, manifest: m as Manifest }
}

/** final_rows 에서 (category, condition, model) 로 한 행을 찾는다. */
export function row(research: Research, category: string, condition: string, model?: string) {
  return research.final_rows.find(
    (r) => r.category === category && r.condition === condition && (model ? r.model === model : true),
  )
}

// ---------------------------------------------------------------- 설명 텍스트 (숫자 없음)

export const PIPELINE = [
  { n: '01', t: '3-view X-ray', d: '0°, 45°, 90° 세 방향에서 촬영한 영상을 입력으로 받는다.' },
  { n: '02', t: 'Adaptive Threshold', d: '영상에 잡음이 있어도 뼈의 윤곽을 안정적으로 찾도록, 배경 잡음 크기를 재서 기준값을 정한다.' },
  { n: '03', t: 'Contour / SDF', d: '찾아낸 뼈 윤곽을, 각 지점이 윤곽선에서 얼마나 떨어져 있는지를 나타내는 숫자 지도로 바꾼다.' },
  { n: '04', t: 'B0 inference', d: '세 방향의 숫자 지도와 기준점 위치를 한꺼번에 이용해, 평균 뼈 모양을 얼마나 변형할지 계산한다.' },
  { n: '05', t: '3D Femur', d: '계산한 변형을 평균 모양에 적용해 3차원 대퇴골 표면을 만든다.' },
]

export const ALGO_MAIN = [
  { t: 'X-ray 3장 (0°/45°/90°)', d: 'CT에서 만든 DRR 영상. 한 장만으로는 깊이를 알 수 없어 서로 다른 각도가 필요하다.' },
  { t: 'Adaptive threshold (M1)', d: '영상 네 귀퉁이(배경)의 중앙값과 MAD로 잡음 크기를 추정해 문턱값을 정한다. 잡음이 없으면 기존 고정값(0.02)을 그대로 쓴다.' },
  { t: 'Contour', d: '문턱값 위 영역 → 3×3 closing → 구멍 채우기 → 가장 큰 덩어리만 남겨 뼈 외곽선을 얻는다.' },
  { t: 'SDF (signed distance field)', d: '정규화한 2D 좌표계에서 윤곽선까지의 부호 있는 거리. 안쪽은 음수, 바깥쪽은 양수.' },
  { t: 'Landmark-weighted latent inference', d: 'SDF 3블록과 landmark 블록을 각각 정규화해 합친 뒤, 정규화 최소제곱으로 형상 계수를 추정한다 (K, γ 고정).' },
  { t: 'SSM (statistical shape model)', d: 'N27명의 CT로 만든 평균 형상과 주성분. 추정한 계수만큼 평균 형상을 변형한다.' },
  { t: '3D femur mesh', d: '4,911개 정점의 대응 메시. 평가 시 E0 pose로 촬영 좌표계에 배치해 GT와 비교한다.' },
]

export const ALGO_FALLBACK = [
  { t: 'GT + kneeCenter 동시 missing', d: '두 기준점이 함께 관측되지 않은 입력에서만 이 경로로 라우팅한다.' },
  { t: 'SDF 3블록 + femurHead 6열', d: '없는 기준점을 평균값으로 채워 넣는 대신, 관측식에서 해당 열을 제거한다.' },
  { t: 'F1 (K=20, γ=0.01)', d: '잠금된 B0의 형상 prior를 그대로 쓰고 관측 구성만 바꾼 제한적 fallback.' },
  { t: '3D femur mesh', d: '평균 오차는 줄지만 꼬리 오차(p95)는 악화가 남아 있다.' },
]

export interface TimelineItem { step: string; title: string; note: string; key?: boolean; tag?: 'adopted' | 'rejected' | 'limit' }

export const TIMELINE: TimelineItem[] = [
  { step: 'Phase 5–6', title: 'SSM · correspondence', note: '해부학 좌표계, nrICP 대응, 크기 정규화로 통계 형상 모델 구축' },
  { step: 'STEP 7', title: 'E0 baseline', note: '2D–3D reconstruction 파이프라인 고정 → 이후 모든 비교의 기준선', key: true },
  { step: 'STEP 8–16', title: 'View / basis 탐색', note: 'GPA, 고차 PC, edge, pose prior, view 개수, 좌우 mirror, 영역 basis' },
  { step: 'STEP 17–22', title: 'Regression · CNN · synthetic', note: 'Ridge/PLS/MLP/CNN/heatmap — domain gap으로 실패', tag: 'rejected' },
  { step: 'STEP 23', title: 'Joint 2D–3D SSM', note: '결합 PCA 시도 실패', tag: 'rejected' },
  { step: 'STEP 24', title: 'Landmark block 가중 재설계', note: '27명 중 25명 개선 — 이 연구의 핵심 전환점', key: true, tag: 'adopted' },
  { step: 'STEP 26', title: 'B0 model lock', note: 'N27 학습, K=15, γ=0.001로 고정하고 독립 6명에 1회 검증', key: true, tag: 'adopted' },
  { step: 'STEP 27–28', title: 'Pat095 · volume bias · missing 분석', note: '원인 진단은 됐으나 개선 방법은 모두 실패', tag: 'rejected' },
  { step: 'STEP 29', title: 'Fallback 후보 개발', note: 'GT+kneeCenter 동시 missing 전용 관측 구성 설계' },
  { step: 'STEP 30–31', title: '외부 데이터 확보 시도', note: 'HFValid 적격 0명, VSDFullBody는 landmark 정의 불일치로 부적격', tag: 'limit' },
  { step: 'STEP 32', title: 'Angle robustness', note: '±1°는 안전, ±2°부터 저하, ±5°에서 열화', tag: 'limit' },
  { step: 'STEP 33', title: 'Blur · contrast · noise', note: 'contrast는 강건, blur medium 열화, Gaussian noise에서 붕괴 발견', tag: 'limit' },
  { step: 'STEP 34', title: 'Noise 붕괴 경계 진단', note: 'SNR을 낮춰가며 어디서 무너지는지, 원인이 contour인지 추적', key: true },
  { step: 'STEP 35', title: 'Adaptive threshold (M1)', note: '배경 MAD 기반 문턱값으로 noise 취약점을 직접 해결', key: true, tag: 'adopted' },
  { step: 'STEP 36', title: 'M1 최종 Main 확정', note: 'clean·blur·contrast·angle 전 조건에서 부작용 없음을 재확인', key: true, tag: 'adopted' },
  { step: 'STEP 37', title: 'F1 fallback 독립 검증', note: '평균은 개선, p95와 hard case는 악화 — 제한적 채택', tag: 'limit' },
  { step: 'STEP 39', title: 'Fallback 개선 마지막 시도', note: 'geometry feature 추가 → 기존 F1보다 나빠 미채택', tag: 'rejected' },
  { step: 'STEP 38', title: 'Final lock', note: 'Main = M1, Fallback = F1로 최종 고정하고 실험 종료', key: true, tag: 'adopted' },
]

export const LIMITATIONS = [
  { t: '독립 검증 대상이 6명뿐이다', d: '모든 최종 수치는 Pat019/031/042/060/080/095 6명에서만 나온 것이다. 통계적 신뢰구간이 넓고, 새로운 집단에서의 성능은 알 수 없다.' },
  { t: '실제 X-ray로 검증하지 않았다', d: '입력은 전부 CT에서 만든 DRR(디지털 재구성 방사선영상)이다. 실제 촬영 영상의 산란·후처리·기기 차이는 반영되지 않았다.' },
  { t: 'CT → DRR 환경의 landmark 가정', d: '2D 기준점은 영상에서 검출한 것이 아니라 3D 좌표를 투영한 값이다. 실제 검출 오차는 이 평가에 포함되어 있지 않다.' },
  { t: '±2° 이상의 촬영각 오차에서 성능이 떨어진다', d: '±1°까지는 안정적이지만 ±2°부터 저하가 시작되고 ±5°에서는 뚜렷하게 열화된다. 촬영 기하가 정확해야 한다.' },
  { t: 'F1 fallback의 p95 / hard case 한계', d: '평균 표면 오차는 줄지만 상위 5% 오차는 오히려 커진다. 개선 시도는 STEP39에서 실패했고 더 진행하지 않았다.' },
  { t: 'Pat095 같은 대상별 hard case', d: '특정 대상에서는 fallback이 오히려 나빠진다. 어떤 대상이 어려운지 미리 알 방법은 아직 없다.' },
  { t: '모든 missing 조합이 검증되지 않았다', d: 'GT+kneeCenter 동시 missing 외의 결측 패턴은 검증 대상이 아니었다. 그 경우 동작은 "판정 불가"로 남는다.' },
  { t: 'volume bias가 남아 있다', d: '복원된 뼈의 부피가 체계적으로 작게 나오는 경향이 있다. STEP28의 보정 시도는 실패했다.' },
  { t: '병리적·이형 대퇴골은 검증하지 않았다', d: '학습 데이터는 정상 형태 중심이다. 골절·변형·인공관절이 있는 경우의 일반화는 확인되지 않았다.' },
]

export const PAGES = [
  { path: '/reconstruction', label: 'RECONSTRUCTION' },
  { path: '/results', label: 'RESULTS' },
  { path: '/robustness', label: 'ROBUSTNESS' },
  { path: '/algorithm', label: 'ALGORITHM' },
  { path: '/limitations', label: 'LIMITATIONS' },
]

/** 핵심 milestone 만 (STEP 전체 나열하지 않음) */
export const MILESTONES = [
  { step: 'PHASE 5–6', title: 'SSM FOUNDATION', note: '해부학 좌표계 · nrICP 대응 · 크기 정규화' },
  { step: 'STEP 7', title: 'E0 BASELINE', note: '2D–3D 파이프라인 고정, 이후 모든 비교의 기준' , key: true },
  { step: 'STEP 24', title: 'LANDMARK-WEIGHTED MODEL', note: '학습 27명 중 25명 개선 — 핵심 전환점', key: true },
  { step: 'STEP 26', title: 'B0 LOCK', note: 'N27 · K=15 · γ=0.001 고정 후 독립 6명 1회 검증', key: true },
  { step: 'STEP 34', title: 'NOISE FAILURE ANALYSIS', note: '잡음이 contour → SDF → latent로 전달돼 붕괴함을 확인' },
  { step: 'STEP 35', title: 'ADAPTIVE THRESHOLD', note: '배경 MAD 기반 문턱값으로 취약점 제거', key: true },
  { step: 'STEP 36', title: 'M1 LOCK', note: 'clean·blur·contrast·angle 전 조건 부작용 없음 확인', key: true },
  { step: 'STEP 37', title: 'F1 VALIDATION', note: '결측 조건 평균 개선, 꼬리 오차 한계 확인' },
  { step: 'STEP 39', title: 'FALLBACK PROBE', note: 'geometry feature 추가 시도 → 기존 F1보다 나빠 미채택' },
  { step: 'STEP 38', title: 'FINAL LOCK', note: 'Main = M1 / Fallback = F1 고정, 실험 종료', key: true },
]
