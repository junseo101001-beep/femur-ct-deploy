import { useEffect, useRef, useState, type ReactNode } from 'react'
import { DEMO_BASE, type CaseData, type Metric } from '../data/research'
import { api } from '../data/api'
import { BONE, FemurViewer } from './FemurViewer'
import { fmt } from './ui'

/* ------------------------------------------------------------------ trace data (same shape for demo + live) */

type Mode = 'clean' | 'missing' | 'fallback'
type LM = 'femurHead' | 'greatTroch' | 'kneeCenter'
const LMS: LM[] = ['femurHead', 'greatTroch', 'kneeCenter']

interface TraceView {
  tag: string
  angle_deg: number
  threshold: { value: number; bg_median: number; bg_mad_scale: number; fixed: number; adaptive_used: boolean; raw_min: number; raw_max: number }
  mask: { url: string; px: number; raw_components: number }
  contour: { n: number; px: [number, number][] }
  sdf: { shape: [number, number]; min: number; max: number; values: number[][]; contour_norm: [number, number][] }
  landmarks: Record<LM, { px: [number, number]; mm: [number, number]; norm: [number, number] }>
  projected_length_mm: number
}
interface TraceCondition {
  condition: Mode
  lmk_used: (number | null)[]
  gkc_removed: boolean
  gkc_replaced_by_mean: boolean
  K: number
  gamma: number
  z: number[]
  z_over_sigma: number[]
  ls_condition_number: number
  shape_delta_mm: { mean: number; p95: number; max: number }
}
interface TraceMeta {
  pid: string
  detector_px: [number, number]
  pix_mm: number
  sdf_grid: { u: [number, number, number]; v: [number, number, number] }
  qc: { pass: boolean }
  landmark_source: string
  threshold_rule: string
}
interface Trace {
  source: 'precomputed' | 'live'
  meta: TraceMeta
  views: TraceView[]
  cond: TraceCondition
  finalMesh: string
  meanMesh: string
  maskUrl: (v: TraceView) => string
  timing?: { preprocess: number; inference: number; total: number }
}

/* ------------------------------------------------------------------ steps */

const STEPS = [
  { n: '01', en: 'X-RAY INPUT', ko: '입력 영상', short: 'X-ray' },
  { n: '02', en: 'ADAPTIVE THRESHOLD', ko: '적응형 임계값', short: 'Threshold' },
  { n: '03', en: 'CONTOUR', ko: '윤곽선 추출', short: 'Contour' },
  { n: '04', en: 'SDF', ko: '부호 거리장', short: 'SDF' },
  { n: '05', en: 'LANDMARK INFERENCE', ko: '기준점 가중 추론 입력', short: 'Landmark' },
  { n: '06', en: 'SSM SHAPE', ko: '통계 형상 모델 추정', short: 'SSM' },
  { n: '07', en: '3D RECONSTRUCTION', ko: '최종 3차원 복원', short: '3D Femur' },
]

const LM_COLOR: Record<LM, string> = { femurHead: '#167a63', greatTroch: '#b88232', kneeCenter: '#5b7896' }

function useImage(url: string | null) {
  const [img, setImg] = useState<HTMLImageElement | null>(null)
  useEffect(() => {
    if (!url) { setImg(null); return }
    let alive = true
    const im = new Image()
    im.onload = () => alive && setImg(im)
    im.src = url
    return () => { alive = false }
  }, [url])
  return img
}

/** 450×600 detector 좌표계 캔버스 — 실제 배열/좌표만 그린다 */
function DetectorCanvas({ drr, mask, draw, label }: {
  drr: string; mask?: string; label: ReactNode
  draw?: (ctx: CanvasRenderingContext2D, maskImg: HTMLImageElement | null) => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const d = useImage(drr)
  const m = useImage(mask ?? null)
  useEffect(() => {
    const cv = ref.current
    if (!cv || !d) return
    const ctx = cv.getContext('2d')!
    ctx.clearRect(0, 0, 450, 600)
    ctx.globalAlpha = 1
    ctx.drawImage(d, 0, 0, 450, 600)
    draw?.(ctx, m)
  }, [d, m, draw])
  return (
    <figure className="pt-cell">
      <canvas ref={ref} width={450} height={600} />
      <figcaption>{label}</figcaption>
    </figure>
  )
}

function tintMask(ctx: CanvasRenderingContext2D, m: HTMLImageElement | null, alpha: number) {
  if (!m) return
  const off = document.createElement('canvas')
  off.width = 450; off.height = 600
  const o = off.getContext('2d')!
  o.drawImage(m, 0, 0)
  const id = o.getImageData(0, 0, 450, 600)
  for (let i = 0; i < id.data.length; i += 4) {
    const on = id.data[i] > 127
    id.data[i] = 22; id.data[i + 1] = 122; id.data[i + 2] = 99; id.data[i + 3] = on ? Math.round(alpha * 255) : 0
  }
  o.putImageData(id, 0, 0)
  ctx.drawImage(off, 0, 0)
}

function polyline(ctx: CanvasRenderingContext2D, pts: [number, number][], color: string, width: number, close = true) {
  ctx.beginPath()
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
  if (close) ctx.closePath()
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'
  ctx.stroke()
}

function SdfCanvas({ v, grid }: { v: TraceView; grid: TraceMeta['sdf_grid'] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const S = 6
  const [rows, cols] = v.sdf.shape
  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(cols, rows)
    const lerp = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t)
    const ivory = [243, 240, 232], green = [22, 122, 99], navy = [23, 34, 53]
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = v.sdf.values[r][c]
        // v_grid 는 아래(-)→위(+) 순서 → 화면은 위가 +v
        const y = rows - 1 - r
        const col = val < 0 ? lerp(ivory, green, Math.pow(Math.min(1, val / v.sdf.min), 0.55)) : lerp(ivory, navy, Math.pow(Math.min(1, val / v.sdf.max), 0.7))
        const i = (y * cols + c) * 4
        img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = 255
      }
    }
    const off = document.createElement('canvas')
    off.width = cols; off.height = rows
    off.getContext('2d')!.putImageData(img, 0, 0)
    ctx.imageSmoothingEnabled = false
    ctx.clearRect(0, 0, cols * S, rows * S)
    ctx.drawImage(off, 0, 0, cols * S, rows * S)
    // 정규화 윤곽 (zero level) — 실제 Q
    const [u0, u1] = grid.u, [v0, v1] = grid.v
    const du = (u1 - u0) / (cols - 1), dv = (v1 - v0) / (rows - 1)
    const toPx = ([u, vv]: [number, number]): [number, number] => [((u - u0) / du + 0.5) * S, (rows - 1 - (vv - v0) / dv + 0.5) * S]
    polyline(ctx, v.sdf.contour_norm.map(toPx), '#ffffff', 2.2)
    polyline(ctx, v.sdf.contour_norm.map(toPx), '#172235', 0.9)
    LMS.forEach((k) => {
      const [x, y] = toPx(v.landmarks[k].norm)
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fillStyle = LM_COLOR[k]; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke()
    })
  }, [v, grid, rows, cols])
  return (
    <figure className="pt-cell sdf">
      <canvas ref={ref} width={cols * S} height={rows * S} />
      <figcaption>
        <span>{v.angle_deg}°</span>
        <span className="mono">{fmt(v.sdf.min, 3)} … {fmt(v.sdf.max, 3)}</span>
      </figcaption>
    </figure>
  )
}

function ZBars({ z }: { z: number[] }) {
  const m = Math.max(3, ...z.map((x) => Math.abs(x)))
  return (
    <div className="pt-z">
      {z.map((x, i) => (
        <div className="pt-z-row" key={i}>
          <span className="mono">z{i + 1}</span>
          <div className="pt-z-track">
            <i className="pt-z-mid" />
            <b className={x < 0 ? 'neg' : 'pos'} style={{ width: `${(Math.abs(x) / m) * 50}%`, [x < 0 ? 'right' : 'left']: '50%' } as React.CSSProperties} />
          </div>
          <span className="mono pt-z-v">{x >= 0 ? '+' : '−'}{Math.abs(x).toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ main component */

const MODE_OPTS: { k: Mode; label: string; ko: string }[] = [
  { k: 'clean', label: 'M1 / normal', ko: '최종 복원' },
  { k: 'missing', label: 'Main / GT+KC missing', ko: '기본 경로 · 누락' },
  { k: 'fallback', label: 'F1 / GT+KC missing', ko: '대체 복원 · 누락' },
]

export function ProcessTrace({ c, caseLabel }: { c: CaseData; caseLabel: string }) {
  // ?step=1..7 로 특정 단계를 바로 열 수 있다 (공유·캡처용)
  const [step, setStep] = useState(() => {
    const q = Number(new URLSearchParams(window.location.search).get('step'))
    return q >= 1 && q <= 7 ? q - 1 : 0
  })
  const [mode, setMode] = useState<Mode>('clean')
  const [playing, setPlaying] = useState(false)
  const [demo, setDemo] = useState<any | null>(null)
  const [live, setLive] = useState<Trace | null>(null)
  const [liveAvailable, setLiveAvailable] = useState(false)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const go = (i: number) => { setStep(Math.max(0, Math.min(STEPS.length - 1, i))); setPlaying(false) }
  const metric: Metric = mode === 'clean' ? c.metrics.clean_m1 : mode === 'missing' ? c.metrics.missing_main : c.metrics.fallback_f1

  // /algorithm#process 로 들어오면 이 영역으로 스크롤
  useEffect(() => {
    if (window.location.hash !== '#process') return
    const t = setTimeout(() => document.getElementById('process')?.scrollIntoView({ block: 'start' }), 400)
    return () => clearTimeout(t)
  }, [])

  // LIVE BACKEND 여부
  useEffect(() => {
    let alive = true
    fetch(api('/api/health'), { signal: AbortSignal.timeout(2500) })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => alive && setLiveAvailable(!!j?.live_inference))
      .catch(() => alive && setLiveAvailable(false))
    return () => { alive = false }
  }, [])

  // demo trace : 케이스가 바뀌면 다시 읽는다
  useEffect(() => {
    let alive = true
    setDemo(null); setLive(null); setErr(null)
    fetch(`${DEMO_BASE}/process/${c.pid}/trace.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`trace.json ${r.status}`))))
      .then((j) => alive && setDemo(j))
      .catch((e: Error) => alive && setErr(e.message))
    return () => { alive = false }
  }, [c.pid])
  useEffect(() => { setLive(null) }, [mode])

  // auto play : 3.2 s 간격, 07 에서 정지
  useEffect(() => {
    if (!playing) return
    if (step >= STEPS.length - 1) { setPlaying(false); return }
    const t = setTimeout(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), 3200)
    return () => clearTimeout(t)
  }, [playing, step])

  const runLive = async () => {
    setBusy(true); setErr(null)
    try {
      const r = await fetch(api('/api/reconstruct/trace'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: c.pid, condition: mode }),
      })
      if (!r.ok) throw new Error(`trace ${r.status}`)
      const j = await r.json()
      setLive({
        source: 'live', meta: j.meta, views: j.views, cond: j.condition,
        finalMesh: api(j.files.final_mesh), meanMesh: api(j.files.mean_shape_mesh),
        maskUrl: (v) => api(v.mask.url), timing: j.timing_ms,
      })
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const T: Trace | null = live ?? (demo ? {
    source: 'precomputed', meta: demo.meta, views: demo.views, cond: demo.conditions[mode],
    finalMesh: `${DEMO_BASE}/${demo.conditions[mode].final_mesh}`, meanMesh: `${DEMO_BASE}/${demo.files.mean_shape_mesh}`,
    maskUrl: (v) => `${DEMO_BASE}/${v.mask.url}`,
  } : null)

  const drr = (i: number) => `${DEMO_BASE}/${c.drr[['000', '045', '090'][i]].file}`
  const removed = (k: LM) => k !== 'femurHead' && !!T && (T.cond.gkc_removed || T.cond.gkc_replaced_by_mean)

  return (
    <section className="pt embedded" id="process">
      <header className="pt-head">
        <div>
          <div className="pt-lab"><span className="pt-num">04</span>RECONSTRUCTION PROCESS <span className="pt-lab-ko">복원 과정</span></div>
          <h2 className="pt-title">X-ray 3장에서 3D 대퇴골까지</h2>
          <p className="pt-sub">
            잠금된 연구 파이프라인이 실제로 계산한 중간 결과를 단계별로 보여준다. 이미지·좌표·계수는 모두 같은 추론 1회에서 수집한 값이다.
          </p>
        </div>
        <div className="pt-src">
          <span className={`pt-badge ${T?.source === 'live' ? 'live' : ''}`}>
            {T?.source === 'live' ? `LIVE TRACE · ${T.timing?.total ?? '—'} ms` : 'PRECOMPUTED TRACE'}
          </span>
          <span className="pt-src-note">{caseLabel}</span>
          {liveAvailable && (
            <button className="rc-opt" disabled={busy} onClick={runLive}>{busy ? 'tracing…' : 'LIVE TRACE 실행'}</button>
          )}
        </div>
      </header>

      <div className="pt-modes">
        <span className="pt-modes-l">CONDITION <em>복원 조건</em></span>
        {MODE_OPTS.map((o) => (
          <button key={o.k} className={`pt-mode${mode === o.k ? ' on' : ''}`} onClick={() => setMode(o.k)}>
            <span className="mono">{o.label}</span><em>{o.ko}</em>
          </button>
        ))}
      </div>

      {/* flow strip — 전체 구조가 가장 먼저 보이도록 */}
      <nav className="pt-flow" aria-label="pipeline">
        {STEPS.map((s, i) => (
          <button key={s.n} className={`pt-flow-i${i === step ? ' on' : ''}${i < step ? ' done' : ''}`} onClick={() => go(i)}>
            <span className="mono">{s.n}</span>{s.short}
          </button>
        ))}
      </nav>

      <div className="pt-grid">
        {/* ------------------------------ left : step list */}
        <ol className="pt-steps">
          {STEPS.map((s, i) => (
            <li key={s.n}>
              <button className={`pt-step${i === step ? ' on' : ''}`} onClick={() => go(i)}>
                <span className="pt-step-n">{s.n}</span>
                <span className="pt-step-t">{s.en}<em>{s.ko}</em></span>
              </button>
              {i < STEPS.length - 1 && <span className="pt-arrow" aria-hidden>↓</span>}
            </li>
          ))}
          <div className="pt-ctrl">
            <button className="rc-opt" disabled={step === 0} onClick={() => go(step - 1)}>PREVIOUS</button>
            <button className="rc-opt" disabled={step === STEPS.length - 1} onClick={() => go(step + 1)}>NEXT</button>
            <button className={`rc-opt${playing ? ' on' : ''}`} onClick={() => { if (step === STEPS.length - 1) setStep(0); setPlaying(!playing) }}>
              {playing ? '❚❚ PAUSE' : '▶ AUTO PLAY'}
            </button>
          </div>
        </ol>

        {/* ------------------------------ center : visualization */}
        <div className="pt-stage">
          <div className="pt-stage-h">
            <span className="mono">{STEPS[step].n}</span> {STEPS[step].en} <em>{STEPS[step].ko}</em>
          </div>
          {!T && <div className="pt-empty">{err ? `trace unavailable — ${err}` : 'loading trace…'}</div>}
          {T && step <= 4 && (
            <div className="pt-views">
              {T.views.map((v, i) => {
                const label = <><span>{v.angle_deg}°</span></>
                if (step === 0) return <DetectorCanvas key={v.tag} drr={drr(i)} label={label} />
                if (step === 1) return (
                  <DetectorCanvas key={v.tag} drr={drr(i)} mask={T.maskUrl(v)} label={<><span>{v.angle_deg}°</span><span className="mono">T = {v.threshold.value.toFixed(4)}</span></>}
                    draw={(ctx, m) => { ctx.fillStyle = 'rgba(5,6,7,0.35)'; ctx.fillRect(0, 0, 450, 600); tintMask(ctx, m, 0.72) }} />
                )
                if (step === 2) return (
                  <DetectorCanvas key={v.tag} drr={drr(i)} mask={T.maskUrl(v)} label={<><span>{v.angle_deg}°</span><span className="mono">{v.contour.n} pts</span></>}
                    draw={(ctx, m) => {
                      ctx.fillStyle = 'rgba(5,6,7,0.45)'; ctx.fillRect(0, 0, 450, 600); tintMask(ctx, m, 0.22)
                      polyline(ctx, v.contour.px, '#7fd3b9', 2.2)
                      ctx.fillStyle = '#ffffff'
                      v.contour.px.forEach(([x, y], k) => { if (k % 12 === 0) { ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill() } })
                    }} />
                )
                if (step === 3) return <SdfCanvas key={v.tag} v={v} grid={T.meta.sdf_grid} />
                return (
                  <DetectorCanvas key={v.tag} drr={drr(i)} label={label}
                    draw={(ctx) => {
                      ctx.fillStyle = 'rgba(5,6,7,0.35)'; ctx.fillRect(0, 0, 450, 600)
                      polyline(ctx, v.contour.px, 'rgba(255,255,255,0.55)', 1.4)
                      LMS.forEach((k) => {
                        const [x, y] = v.landmarks[k].px
                        const off = removed(k)
                        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2)
                        ctx.fillStyle = off ? 'rgba(0,0,0,0)' : LM_COLOR[k]; ctx.fill()
                        ctx.strokeStyle = off ? '#b96d78' : '#ffffff'; ctx.lineWidth = 2.5; ctx.stroke()
                        if (off) { ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke() }
                        ctx.font = '600 17px "IBM Plex Mono", monospace'; ctx.fillStyle = '#ffffff'
                        ctx.fillText(k === 'femurHead' ? 'FH' : k === 'greatTroch' ? 'GT' : 'KC', x + 13, y + 6)
                      })
                    }} />
                )
              })}
            </div>
          )}
          {T && step === 5 && (
            <div className="pt-ssm">
              <div className="pt-ssm-v">
                <figure>
                  <FemurViewer className="pt-viewer" height={420} autoRotate={false} background={null} layers={[{ url: T.meanMesh, color: '#b9b3a6', opacity: 1, visible: true }]} />
                  <figcaption>SSM mean shape <em>평균 형상 · 같은 E0 pose</em></figcaption>
                </figure>
                <span className="pt-ssm-arrow">→</span>
                <figure>
                  <FemurViewer className="pt-viewer" height={420} autoRotate={false} background={null} layers={[{ url: T.finalMesh, color: BONE, opacity: 1, visible: true }]} />
                  <figcaption>estimated shape <em>추정 계수 적용 결과</em></figcaption>
                </figure>
              </div>
            </div>
          )}
          {T && step === 6 && (
            <div className="pt-final">
              <FemurViewer className="pt-viewer" height={520} autoRotate background={null} layers={[{ url: T.finalMesh, color: BONE, opacity: 1, visible: true }]} />
            </div>
          )}
        </div>

        {/* ------------------------------ right : description + actual values */}
        <aside className="pt-info">
          {T && <StepInfo step={step} T={T} metric={metric} />}
        </aside>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ right panel content per step */

function KV({ k, v }: { k: ReactNode; v: ReactNode }) {
  return <div className="pt-kv"><span>{k}</span><b>{v}</b></div>
}

function StepInfo({ step, T, metric }: { step: number; T: Trace; metric: Metric }) {
  const V = T.views
  const table = (head: string[], rows: ReactNode[][]) => (
    <table className="pt-table">
      <thead><tr>{head.map((h) => <th key={h}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((x, j) => <td key={j}>{x}</td>)}</tr>)}</tbody>
    </table>
  )
  switch (step) {
    case 0:
      return (<>
        <h3>X-ray input</h3>
        <p>CT에서 생성한 DRR 3장(0°·45°·90°)을 입력으로 사용한다. 실제 촬영 X-ray가 아니다.</p>
        <KV k="Detector" v={`${T.meta.detector_px[0]} × ${T.meta.detector_px[1]} px`} />
        <KV k="Pixel spacing" v={`${T.meta.pix_mm} mm`} />
        {table(['view', 'raw min', 'raw max'], V.map((v) => [`${v.angle_deg}°`, fmt(v.threshold.raw_min, 3), fmt(v.threshold.raw_max, 3)]))}
      </>)
    case 1:
      return (<>
        <h3>Adaptive threshold</h3>
        <p>영상 네 귀퉁이(40×40 px) 배경의 중앙값과 MAD로 잡음 크기를 재서 임계값을 정한다. 초록 영역이 임계값을 넘은 실제 mask다.</p>
        <div className="pt-formula mono">T = max(0.02, median + 3 × 1.4826 × MAD)</div>
        {table(['view', 'median', '1.4826·MAD', 'T'], V.map((v) => [`${v.angle_deg}°`, fmt(v.threshold.bg_median, 4), fmt(v.threshold.bg_mad_scale, 4), fmt(v.threshold.value, 4)]))}
        <p className="pt-note">
          {V.every((v) => !v.threshold.adaptive_used)
            ? '이 입력은 잡음 없는 DRR이라 배경 MAD가 0이고, 규칙에 따라 고정값 0.02가 그대로 쓰였다.'
            : '배경 잡음이 있어 고정값 0.02보다 높은 임계값이 적용되었다.'}
        </p>
      </>)
    case 2:
      return (<>
        <h3>Contour extraction</h3>
        <p>임계값 영역 → 3×3 closing → 구멍 채우기 → 가장 큰 연결 영역만 남긴 뒤, 외곽선을 따라 같은 간격 240점으로 다시 샘플링한다.</p>
        {table(['view', 'mask px', 'raw comp.', 'points'], V.map((v) => [`${v.angle_deg}°`, v.mask.px.toLocaleString(), v.mask.raw_components, v.contour.n]))}
      </>)
    case 3:
      return (<>
        <h3>SDF representation</h3>
        <p>윤곽선을 머리 방향 기준의 정규화 좌표계로 옮기고, 격자 각 점에서 윤곽선까지의 부호 있는 거리를 계산한다. 안쪽은 음수(초록), 바깥은 양수(남색).</p>
        <KV k="Grid" v={`${T.meta.sdf_grid.v[2]} × ${T.meta.sdf_grid.u[2]}`} />
        <KV k="u range" v={`${T.meta.sdf_grid.u[0]} … ${T.meta.sdf_grid.u[1]}`} />
        <KV k="v range" v={`${T.meta.sdf_grid.v[0]} … ${T.meta.sdf_grid.v[1]}`} />
        {table(['view', 'min', 'max', 'length mm'], V.map((v) => [`${v.angle_deg}°`, fmt(v.sdf.min, 3), fmt(v.sdf.max, 3), fmt(v.projected_length_mm, 1)]))}
        <p className="pt-note">색 점은 같은 정규화 좌표계의 landmark 위치 (FH 초록 · GT 황색 · KC 청색).</p>
      </>)
    case 4:
      return (<>
        <h3>Landmark-weighted inference</h3>
        <p>SDF 3블록(각 112×40)과 landmark 블록(3점 × 2좌표 × 3뷰 = 18)을 각각 정규화해 하나의 관측 벡터로 합친다.</p>
        {table(['view', 'FH (px)', 'GT (px)', 'KC (px)'], V.map((v) => [`${v.angle_deg}°`, ...LMS.map((k) => `${v.landmarks[k].px[0].toFixed(0)}, ${v.landmarks[k].px[1].toFixed(0)}`)]))}
        {T.cond.gkc_removed && <p className="pt-note warn">F1: GT·KC 12열을 관측식에서 제거하고 SDF + femurHead만 사용한다.</p>}
        {T.cond.gkc_replaced_by_mean && <p className="pt-note warn">Main 경로(결측): GT·KC 12열을 training 평균값으로 채운다.</p>}
        <div className="pt-warn">
          <b>Landmark source</b>
          현재 연구 평가에서는 3D ground-truth landmark를 2D 영상 좌표로 투영하여 사용한다. X-ray에서 자동 검출한 landmark가 아니다.
          <span className="mono">3D landmark projection used for current research evaluation</span>
        </div>
      </>)
    case 5:
      return (<>
        <h3>SSM shape estimation</h3>
        <p>정규화 최소제곱으로 K개 형상 계수 z를 한 번에 풀고, 그 계수로 SSM 평균 형상을 변형한다. 왼쪽은 평균 형상, 오른쪽은 추정 결과다 (보간 애니메이션 없음).</p>
        <KV k="K" v={T.cond.K} />
        <KV k="γ" v={T.cond.gamma} />
        <KV k="Condition number" v={T.cond.ls_condition_number.toExponential(2)} />
        <div className="pt-mini-lab" style={{ marginTop: 18 }}>ESTIMATED − MEAN (대응 정점 거리)</div>
        <KV k="mean" v={`${fmt(T.cond.shape_delta_mm.mean, 3)} mm`} />
        <KV k="p95" v={`${fmt(T.cond.shape_delta_mm.p95, 3)} mm`} />
        <KV k="max" v={`${fmt(T.cond.shape_delta_mm.max, 3)} mm`} />
        <div className="pt-mini-lab" style={{ marginTop: 18 }}>LATENT COEFFICIENTS z<sub>k</sub> / σ<sub>k</sub></div>
        <ZBars z={T.cond.z_over_sigma} />
      </>)
    default:
      return (<>
        <h3>3D reconstruction</h3>
        <p>추정 형상을 평가와 같은 E0 pose로 촬영 좌표계에 배치한 최종 메시 (4,911 정점). 아래 값은 연구 검증 결과다.</p>
        <div className="pt-metrics">
          <KV k={<>Symmetric surface error<em>대칭 표면 오차</em></>} v={`${fmt(metric.sym, 3)} mm`} />
          <KV k={<>P95<em>95백분위 표면 오차</em></>} v={`${fmt(metric.p95, 3)} mm`} />
          <KV k={<>Cov5<em>5 mm 이내 포함률</em></>} v={`${fmt(metric.cov5, 2)} %`} />
        </div>
        <p className="pt-note">최종 메시는 기존 /api/reconstruct 결과 및 연구 저장 prediction과 정점 단위로 동일함을 확인했다.</p>
      </>)
  }
}
