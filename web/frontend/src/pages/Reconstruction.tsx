import { useEffect, useState, type ReactNode } from 'react'
import { Gate } from '../data/store'
import { DEMO_BASE, caseLabel, type CaseData, type Metric } from '../data/research'
import { api } from '../data/api'
import { BONE, BONE_ALT, FemurViewer, GT_COLOR, initialViewerState, type Layer, type ViewerState } from '../components/FemurViewer'
import { fmt, signed } from '../components/ui'

type Mode = 'clean' | 'missing' | 'fallback'
interface RunState { status: 'idle' | 'running' | 'done'; stage: string; source: string | null; serverMs: number | null }
interface HealthInfo { live_inference: boolean; model_md5?: string; mode?: string }

const MODES: { k: Mode; label: string; ko: string; mesh: string; color: string; desc: string }[] = [
  { k: 'clean', label: 'M1 / normal', ko: '최종 복원', mesh: 'recon_clean', color: BONE, desc: '모든 기준점이 관측된 정상 입력. 최종 Main 모델 M1을 사용한다.' },
  { k: 'missing', label: 'Main / GT+KC missing', ko: '기본 경로 · GT·KC 누락', mesh: 'recon_missing_main', color: BONE_ALT, desc: 'GT와 kneeCenter가 동시에 없는 입력을 Main 경로가 그대로 처리한 경우 (fallback 미적용).' },
  { k: 'fallback', label: 'F1 / GT+KC missing', ko: '대체 복원 · GT·KC 누락', mesh: 'recon_fallback_f1', color: '#c2b7a4', desc: '같은 결측 입력을 F1 fallback으로 복원한 경우.' },
]

const IDLE: RunState = { status: 'idle', stage: '', source: null, serverMs: null }
const VIEWS = ['000', '045', '090'] as const

/* ------------------------------------------------------------------ X-ray popup */

function XrayModal({ c, idx, caseText, onNav, onClose }: { c: CaseData; idx: number; caseText: string; onNav: (i: number) => void; onClose: () => void }) {
  const d = c.drr[VIEWS[idx]]
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight') onNav((idx + 1) % VIEWS.length)
      if (e.key === 'ArrowLeft') onNav((idx + VIEWS.length - 1) % VIEWS.length)
    }
    window.addEventListener('keydown', onKey)
    return () => { document.body.style.overflow = prev; window.removeEventListener('keydown', onKey) }
  }, [idx, onNav, onClose])

  return (
    <div className="xm" role="dialog" aria-modal="true" aria-label={`DRR ${d.angle_deg}°`} onClick={onClose}>
      <div className="xm-box" onClick={(e) => e.stopPropagation()}>
        <div className="xm-head">
          <div>
            <div className="xm-lab">X-RAY INPUT · DRR</div>
            <div className="xm-title"><span className="mono">{d.angle_deg}°</span> {caseText}</div>
          </div>
          <button className="xm-close" onClick={onClose} aria-label="닫기">
            <svg width="12" height="12" viewBox="0 0 12 12" stroke="currentColor" strokeWidth="1.5"><path d="M2 2l8 8M10 2 2 10" /></svg>
          </button>
        </div>

        <div className="xm-body">
          <button className="xm-nav" onClick={() => onNav((idx + VIEWS.length - 1) % VIEWS.length)} aria-label="이전 view">‹</button>
          <div className="xm-img"><img src={`${DEMO_BASE}/${d.file}`} alt={`DRR ${d.angle_deg}°`} /></div>
          <button className="xm-nav" onClick={() => onNav((idx + 1) % VIEWS.length)} aria-label="다음 view">›</button>
        </div>

        <div className="xm-foot">
          <div className="xm-thumbs">
            {VIEWS.map((k, i) => (
              <button key={k} className={i === idx ? 'on' : ''} onClick={() => onNav(i)}>
                <img src={`${DEMO_BASE}/${c.drr[k].file}`} alt="" />
                <span>{c.drr[k].angle_deg}°</span>
              </button>
            ))}
          </div>
          <dl className="xm-meta">
            <div><dt>Detector</dt><dd>{d.shape[1]} × {d.shape[0]} px · 0.8 mm</dd></div>
            <div><dt>Raw line integral</dt><dd>{d.raw_min.toFixed(3)} … {d.raw_max.toFixed(3)}</dd></div>
            <div><dt>Source</dt><dd>CT-derived DRR <em>실제 촬영 X-ray 아님</em></dd></div>
          </dl>
        </div>
        <div className="xm-hint">← → view 이동 · Esc 닫기</div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ small parts */

const Ico = {
  case: <path d="M3 4.5h4l1.2 1.5H13v6.5H3z" />,
  model: <><circle cx="8" cy="8" r="5" /><path d="M8 3v10M3 8h10" /></>,
  view: <><path d="M8 2.5 13 5.2v5.6L8 13.5 3 10.8V5.2z" /><path d="M3 5.2 8 8l5-2.8M8 8v5.5" /></>,
  display: <><rect x="3" y="3" width="4" height="4" /><rect x="9" y="3" width="4" height="4" /><rect x="3" y="9" width="4" height="4" /><rect x="9" y="9" width="4" height="4" /></>,
  run: <path d="M5 3.5v9l7.5-4.5z" />,
}

function Section({ icon, title, ko, children, defaultOpen = true }: { icon: ReactNode; title: string; ko?: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="rc-sec">
      <button className="rc-sec-h" onClick={() => setOpen(!open)} aria-expanded={open}>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">{icon}</svg>
        <span className="rc-sec-t">{title}</span>
        {ko && <span className="rc-sec-ko">{ko}</span>}
        <svg className={`rc-chev${open ? ' open' : ''}`} width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.3"><path d="M2 6.5 5 3.5l3 3" /></svg>
      </button>
      {open && <div className="rc-sec-b">{children}</div>}
    </section>
  )
}

function Check({ on, set, children }: { on: boolean; set: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="rc-check">
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />
      <span>{children}</span>
    </label>
  )
}

function Slider({ label, value, min, max, set, disabled }: { label: ReactNode; value: number; min: number; max: number; set: (v: number) => void; disabled?: boolean }) {
  return (
    <div className={`rc-slider${disabled ? ' off' : ''}`}>
      <span className="rc-slider-l">{label}</span>
      <input type="range" min={min} max={max} step={0.01} value={value} disabled={disabled} onChange={(e) => set(Number(e.target.value))} />
      <span className="rc-slider-v">{Math.round(value * 100)}%</span>
    </div>
  )
}

function MetricRow({ en, ko, v, u, tone }: { en: string; ko: string; v: ReactNode; u?: string; tone?: 'good' | 'bad' }) {
  return (
    <div className="rm-row">
      <div>
        <div className="rm-ko">{ko}</div>
        <div className="rm-en">{en}</div>
      </div>
      <div className={`rm-v${tone ? ` ${tone}` : ''}`}>
        {v}{u && <span className="rm-u">{u}</span>}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ page */

export default function Reconstruction() {
  const [caseIdx, setCaseIdx] = useState(0)
  const [mode, setMode] = useState<Mode>('clean')
  const [vs, setVs] = useState<ViewerState>(initialViewerState)
  const [run, setRun] = useState<RunState>(IDLE)
  const [live, setLive] = useState<Partial<Record<Mode, string>>>({})
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [xray, setXray] = useState<number | null>(null)          // 확대 중인 DRR view index
  const set = (p: Partial<ViewerState>) => setVs((s) => ({ ...s, ...p }))

  useEffect(() => {
    let alive = true
    fetch(api('/api/health'), { signal: AbortSignal.timeout(2500) })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => alive && setHealth(j))
      .catch(() => alive && setHealth(null))
    return () => { alive = false }
  }, [])

  const runReconstruction = async (c: CaseData) => {
    const stages = ['read 3 views', 'adaptive threshold', 'contour / SDF', 'latent inference', 'SSM → mesh']
    setRun({ status: 'running', stage: stages[0], source: null, serverMs: null })
    for (const st of stages) {
      setRun((r) => ({ ...r, stage: st }))
      await new Promise((res) => setTimeout(res, 220))
    }
    if (health?.live_inference) {
      try {
        const r = await fetch(api('/api/reconstruct'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ case_id: c.pid, condition: mode }),
        })
        if (r.ok) {
          const j = await r.json()
          if (j.mesh_url) setLive((L) => ({ ...L, [mode]: api(j.mesh_url as string) }))
          setRun({ status: 'done', stage: 'complete', source: 'server (locked pipeline)', serverMs: j.timing_ms?.total ?? null })
          return
        }
      } catch { /* precomputed 로 대체 */ }
    }
    setRun({ status: 'done', stage: 'complete', source: 'precomputed', serverMs: null })
  }

  return (
    <main className="rc-page">
      <Gate>
        {({ manifest, research }) => {
          const c = manifest.cases[caseIdx]
          const active = MODES.find((m) => m.k === mode)!
          const m: Metric = mode === 'clean' ? c.metrics.clean_m1 : mode === 'missing' ? c.metrics.missing_main : c.metrics.fallback_f1
          const meshUrl = live[mode] ?? `${DEMO_BASE}/${c.meshes[active.mesh].file}`
          const layers: Layer[] = [
            { url: meshUrl, color: active.color, opacity: vs.opacity, visible: true, wireframe: vs.wireframe },
            { url: `${DEMO_BASE}/${c.meshes.gt.file}`, color: GT_COLOR, opacity: vs.gtOpacity, visible: vs.showGt, smooth: false },
          ]
          const dE0 = c.metrics.delta_b0_minus_e0
          const lock = research.lock
          const isLive = !!health?.live_inference

          return (
            <div className="rc-grid">
              {/* ============================================== left: control panel */}
              <aside className="rc-panel rc-left">
                <Section icon={Ico.case} title="CASE" ko="검증 대상">
                  <div className="rc-opts cases">
                    {manifest.cases.map((cc, i) => (
                      <button key={cc.id} className={`rc-opt${i === caseIdx ? ' on' : ''}`}
                              onClick={() => { setCaseIdx(i); setRun(IDLE); setLive({}) }}>
                        {caseLabel(cc.pid, i)}
                      </button>
                    ))}
                  </div>
                </Section>

                <Section icon={Ico.model} title="MODEL" ko="복원 조건">
                  <div className="rc-opts">
                    {MODES.map((mm) => (
                      <button key={mm.k} className={`rc-opt model${mode === mm.k ? ' on' : ''}`} onClick={() => setMode(mm.k)}>
                        <span className="en">{mm.label}</span>
                        <span className="ko">{mm.ko}</span>
                      </button>
                    ))}
                  </div>
                  <p className="rc-note">{active.desc}</p>
                </Section>

                <Section icon={Ico.view} title="VIEW" ko="시점">
                  <div className="rc-opts four">
                    {([['front', 'Front'], ['side', 'Side'], ['top', 'Top']] as const).map(([p, label]) => (
                      <button key={p} className={`rc-opt${vs.preset === p ? ' on' : ''}`} onClick={() => set({ preset: p })}>{label}</button>
                    ))}
                    <button className="rc-opt" onClick={() => set({ preset: 'free', resetToken: vs.resetToken + 1 })}>Reset</button>
                  </div>
                </Section>

                <Section icon={Ico.display} title="DISPLAY" ko="표시">
                  <Check on={vs.wireframe} set={(v) => set({ wireframe: v })}>Wireframe <em>와이어프레임</em></Check>
                  <Check on={vs.showGt} set={(v) => set({ showGt: v, opacity: v ? 0.5 : 1 })}>GT overlay <em>정답 중첩 표시</em></Check>
                  <Slider label="GT opacity" value={vs.gtOpacity} min={0.1} max={1} set={(v) => set({ gtOpacity: v })} disabled={!vs.showGt} />
                  <Slider label="Opacity" value={vs.opacity} min={0.15} max={1} set={(v) => set({ opacity: v })} />
                  <Check on={vs.autoRotate} set={(v) => set({ autoRotate: v })}>Auto rotate <em>자동 회전</em></Check>
                </Section>

                <Section icon={Ico.run} title="RUN" ko="복원 실행">
                  <a className="rc-process-link" href="/algorithm#process">복원 과정 단계별로 보기 (알고리즘) →</a>
                  <div className={`rc-mode ${isLive ? 'live' : 'demo'}`}>
                    <span className="dot" />
                    {isLive ? 'LIVE BACKEND · locked M1 pipeline' : 'DEMO MODE · precomputed results'}
                  </div>
                  <div className="rc-opts two" style={{ marginTop: 12 }}>
                    <button className="rc-opt primary" disabled={run.status === 'running'} onClick={() => runReconstruction(c)}>
                      {run.status === 'running' ? '실행 중…' : '복원 실행'}
                    </button>
                    <button className="rc-opt" onClick={() => { setRun(IDLE); setLive({}); setVs(initialViewerState) }}>처음으로</button>
                  </div>
                  {run.status !== 'idle' && (
                    <div className={`rc-run${run.status === 'done' ? ' done' : ''}`}>
                      {run.status === 'running' ? `${run.stage}…` : `${run.stage} — ${run.source}${run.serverMs ? ` · ${run.serverMs} ms` : ''}`}
                    </div>
                  )}
                  {run.status === 'done' && (
                    <p className="rc-note">
                      {run.source === 'precomputed'
                        ? '브라우저에서 추론한 것이 아니라 연구 파이프라인이 미리 계산한 결과다.'
                        : '서버가 잠금 파이프라인(M1 전처리 → B0 추론)을 실제로 실행해 메시를 생성했다.'}
                    </p>
                  )}
                </Section>
              </aside>

              {/* ============================================== center: 3D stage */}
              <div className="rc-stage">
                <FemurViewer
                  key={c.id}
                  className="rc-canvas"
                  height="100%"
                  layers={layers}
                  autoRotate={vs.autoRotate}
                  preset={vs.preset}
                  resetToken={vs.resetToken}
                  background={null}
                  shadow
                  fov={30}
                  axes
                >
                  <div className="rc-stage-tag">
                    <span className="mono">{active.label}</span>
                    <span>{caseLabel(c.pid, caseIdx)}</span>
                    {vs.showGt && <span style={{ color: '#9a7a3a' }}>GT overlay</span>}
                    {live[mode] && <span className="acc">live server mesh</span>}
                  </div>
                  <div className="rc-hint">
                    <svg width="18" height="12" viewBox="0 0 18 12" fill="none" stroke="currentColor" strokeWidth="1.1"><ellipse cx="9" cy="6" rx="8" ry="4.5" /><path d="M13.5 3.2 16 4l-.6-2.6" /></svg>
                    <span>Drag to rotate · Scroll to zoom · Right-drag to pan</span>
                  </div>
                </FemurViewer>

                <div className="rc-xrays">
                  {VIEWS.map((k, vi) => (
                    <figure key={k}>
                      <button className="rc-xray" onClick={() => setXray(vi)} aria-label={`DRR ${c.drr[k].angle_deg}° 크게 보기`}>
                        <img src={`${DEMO_BASE}/${c.drr[k].file}`} alt={`DRR ${c.drr[k].angle_deg}°`} />
                        <span className="rc-xray-zoom" aria-hidden>
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4"><circle cx="5" cy="5" r="3.6" /><path d="M7.7 7.7 11 11M3.5 5h3M5 3.5v3" /></svg>
                        </span>
                      </button>
                      <figcaption><span>{c.drr[k].angle_deg}°</span><i /></figcaption>
                    </figure>
                  ))}
                  <div className="rc-xray-note">DRR input<br />CT-derived · 450×600</div>
                </div>
              </div>

              {/* ============================================== right: result panel */}
              <aside className="rc-panel rc-right">
                <div className="rm-lab">RECONSTRUCTION</div>
                <h1 className="rm-title">3D Femur Reconstruction</h1>
                <div className="rm-sub">다중 뷰 X-ray 기반 대퇴골 3차원 복원</div>
                <div className="rm-badge"><span className="dot" />{lock.main.name} / LOCKED <span className="ko">· 최종 확정 모델</span></div>

                <div className="rm-ctx">
                  <span>{caseLabel(c.pid, caseIdx)}</span>
                  <span>{active.label}</span>
                </div>

                <div className="rm-list">
                  <MetricRow en="Symmetric surface error" ko="평균 대칭 표면 오차" v={fmt(m.sym, 3)} u="mm" />
                  <MetricRow en="P95" ko="95백분위 표면 오차" v={fmt(m.p95, 3)} u="mm" />
                  <MetricRow en="Cov5" ko="5 mm 이내 포함률" v={fmt(m.cov5, 2)} u="%" />
                  <MetricRow en="Volume error" ko="부피 오차" v={m.vol_err_pct == null ? '—' : signed(m.vol_err_pct, 2)} u="%" />
                  <MetricRow en="Latent error (α)" ko="잠재계수 오차" v={fmt(m.alpha_rmse_sigma, 3)} u="σ" />
                </div>

                <div className="rm-group">
                  <div className="rm-glab">BASELINE COMPARISON <span>기존 모델 대비 · M1 정상 입력</span></div>
                  <MetricRow en="Baseline (E0)" ko="기존 모델" v={fmt(c.metrics.baseline_e0.sym, 3)} u="mm" />
                  <MetricRow en="Improvement (M1 − E0)" ko="개선량" v={signed(dE0, 3)} u="mm" tone={dE0 < 0 ? 'good' : 'bad'} />
                  <MetricRow en="Cohort validation" ko="검증 대상 개선" v={`${research.clean.n_better} / 6`} />
                </div>

                <div className="rm-group">
                  <div className="rm-glab">POSE <span>자세 추정 · M1 정상 입력</span></div>
                  <MetricRow en="Pose rotation" ko="자세 회전 오차" v={fmt(c.metrics.clean_m1.rot_deg, 3)} u="°" />
                  <MetricRow en="Pose translation" ko="자세 이동 오차" v={fmt(c.metrics.clean_m1.trans_mm, 3)} u="mm" />
                </div>

                <div className="rm-group">
                  <div className="rm-glab">CONDITION COMPARISON <span>조건별 비교 · mm</span></div>
                  <table className="rm-table">
                    <thead><tr><th>condition</th><th>Sym</th><th>p95</th></tr></thead>
                    <tbody>
                      <tr className={mode === 'clean' ? 'on' : undefined}><td>M1 normal <em>최종 복원</em></td><td>{fmt(c.metrics.clean_m1.sym)}</td><td>{fmt(c.metrics.clean_m1.p95)}</td></tr>
                      <tr className={mode === 'missing' ? 'on' : undefined}><td>Main, GT+KC missing <em>기본 경로</em></td><td>{fmt(c.metrics.missing_main.sym)}</td><td>{fmt(c.metrics.missing_main.p95)}</td></tr>
                      <tr className={mode === 'fallback' ? 'on' : undefined}><td>F1 fallback <em>대체 복원</em></td><td>{fmt(c.metrics.fallback_f1.sym)}</td><td>{fmt(c.metrics.fallback_f1.p95)}</td></tr>
                      <tr><td>E0 baseline <em>기존 모델</em></td><td>{fmt(c.metrics.baseline_e0.sym)}</td><td>{fmt(c.metrics.baseline_e0.p95)}</td></tr>
                    </tbody>
                  </table>
                  <div className="rm-foot">화면 정합은 평가와 동일한 E0 pose 사용</div>
                </div>

                <div className="rm-info">
                  <div className="rm-glab">MODEL INFO</div>
                  <div className="rm-info-row">
                    <div><span>Base</span><b>B0</b></div>
                    <div><span>Training</span><b>N = 27</b></div>
                    <div><span>Latent</span><b>K = {lock.main.b0_K}</b></div>
                    <div><span>Regularization</span><b>γ = {lock.main.b0_gamma}</b></div>
                  </div>
                </div>
              </aside>
              {xray !== null && (
                <XrayModal c={c} idx={xray} caseText={caseLabel(c.pid, caseIdx)} onNav={setXray} onClose={() => setXray(null)} />
              )}
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
