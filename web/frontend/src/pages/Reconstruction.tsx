import { useEffect, useState } from 'react'
import { Gate } from '../data/store'
import { DEMO_BASE, caseLabel, type CaseData } from '../data/research'
import { api } from '../data/api'
import { BONE, BONE_ALT, FemurViewer, GT_COLOR, ViewerControls, initialViewerState, type Layer, type ViewerState } from '../components/FemurViewer'
import { DL, Lab, PageHead, fmt, signed } from '../components/ui'

type Mode = 'clean' | 'missing' | 'fallback'
interface RunState { status: 'idle' | 'running' | 'done'; stage: string; source: string | null; serverMs: number | null }
interface HealthInfo { live_inference: boolean; model_md5?: string; mode?: string }

const MODES: { k: Mode; label: string; mesh: string; color: string; desc: string }[] = [
  { k: 'clean', label: 'M1 / normal', mesh: 'recon_clean', color: BONE, desc: '모든 기준점이 관측된 정상 입력. 최종 Main 모델 M1이 사용됩니다.' },
  { k: 'missing', label: 'Main / GT+KC missing', mesh: 'recon_missing_main', color: BONE_ALT, desc: 'GT와 kneeCenter가 동시에 없는 입력을 Main 경로가 그대로 처리한 경우 (fallback 미적용).' },
  { k: 'fallback', label: 'F1 / GT+KC missing', mesh: 'recon_fallback_f1', color: '#c2b7a4', desc: '같은 결측 입력을 F1 fallback으로 복원한 경우.' },
]

export default function Reconstruction() {
  const [caseIdx, setCaseIdx] = useState(0)
  const [mode, setMode] = useState<Mode>('clean')
  const [vs, setVs] = useState<ViewerState>(initialViewerState)
  const [run, setRun] = useState<RunState>({ status: 'idle', stage: '', source: null, serverMs: null })
  const [live, setLive] = useState<Partial<Record<Mode, string>>>({})
  const [health, setHealth] = useState<HealthInfo | null>(null)
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
    <main className="page">
      <Gate>
        {({ manifest, research }) => {
          const c = manifest.cases[caseIdx]
          const active = MODES.find((m) => m.k === mode)!
          const m = mode === 'clean' ? c.metrics.clean_m1 : mode === 'missing' ? c.metrics.missing_main : c.metrics.fallback_f1
          const meshUrl = live[mode] ?? `${DEMO_BASE}/${c.meshes[active.mesh].file}`
          const layers: Layer[] = [
            { url: meshUrl, color: active.color, opacity: vs.opacity, visible: true, wireframe: vs.wireframe },
            { url: `${DEMO_BASE}/${c.meshes.gt.file}`, color: GT_COLOR, opacity: vs.gtOpacity, visible: vs.showGt, smooth: false },
          ]
          const dE0 = c.metrics.delta_b0_minus_e0

          return (
            <div className="wrap">
              <PageHead
                label="Reconstruction / multi-view input"
                title="3-view 관측에서 3차원 대퇴골 추정"
                note="왼쪽은 연구에 사용된 DRR 3장, 가운데는 복원 결과, 오른쪽은 해당 케이스의 정량 지표입니다. 모든 값은 연구 산출물에서 직접 읽습니다."
              />

              <div className="spread" style={{ paddingBottom: 18, borderBottom: '1px solid var(--line)' }}>
                <div className="row" style={{ gap: 6 }}>
                  <span className="lab" style={{ marginRight: 8 }}>Case</span>
                  {manifest.cases.map((cc, i) => (
                    <button key={cc.id} className={`tbtn ${i === caseIdx ? 'on' : ''}`}
                            onClick={() => { setCaseIdx(i); setRun({ status: 'idle', stage: '', source: null, serverMs: null }); setLive({}) }}>
                      {caseLabel(cc.pid, i)}
                    </button>
                  ))}
                </div>
                <div className="mono tiny" style={{ color: health?.live_inference ? 'var(--accent)' : 'var(--muted)' }}>
                  {health?.live_inference ? 'LIVE BACKEND / locked M1 pipeline' : 'DEMO MODE / precomputed result'}
                </div>
              </div>

              <div className="recon-grid">
                {/* ---------------------------------------------- input */}
                <div>
                  <Lab>3-view input</Lab>
                  <div style={{ display: 'grid', gap: 14, marginTop: 14 }}>
                    {(['000', '045', '090'] as const).map((k) => (
                      <div className="xray" key={k}>
                        <img src={`${DEMO_BASE}/${c.drr[k].file}`} alt={`DRR ${c.drr[k].angle_deg}°`} loading="lazy" />
                        <div className="xray-cap">
                          <span className="mono tiny">{c.drr[k].angle_deg}°</span>
                          <span className="mono tiny dim">{c.drr[k].shape[1]}×{c.drr[k].shape[0]}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mono tiny dim" style={{ marginTop: 16, lineHeight: 1.95 }}>
                    DRR (CT-derived)<br />detector 450×600 px / 0.8 mm<br />SID 1150 / SOD 1050 mm
                  </div>
                </div>

                {/* ---------------------------------------------- viewer */}
                <div>
                  <div className="spread" style={{ marginBottom: 12 }}>
                    <Lab>3D viewer</Lab>
                    <div className="row" style={{ gap: 6 }}>
                      {MODES.map((mm) => (
                        <button key={mm.k} className={`tbtn ${mode === mm.k ? 'on' : ''}`} onClick={() => setMode(mm.k)}>{mm.label}</button>
                      ))}
                    </div>
                  </div>

                  <FemurViewer
                    key={c.id}
                    height={520}
                    layers={layers}
                    autoRotate={vs.autoRotate}
                    preset={vs.preset}
                    resetToken={vs.resetToken}
                    background="#12161a"
                  >
                    <div className="viewer-note">
                      <div className="mono tiny dim">
                        {active.label} · {caseLabel(c.pid, caseIdx)}{vs.showGt ? ' · GT overlay' : ''}{live[mode] ? ' · live server mesh' : ''}
                      </div>
                    </div>
                  </FemurViewer>

                  <ViewerControls s={vs} set={set} />

                  <div className="row" style={{ marginTop: 24, gap: 8, paddingTop: 18, borderTop: '1px solid var(--line)' }}>
                    <button className="tbtn wide" disabled={run.status === 'running'} onClick={() => runReconstruction(c)}>
                      {run.status === 'running' ? 'Running…' : 'Run reconstruction'}
                    </button>
                    <button className="tbtn wide" onClick={() => { setRun({ status: 'idle', stage: '', source: null, serverMs: null }); setLive({}); setVs(initialViewerState) }}>
                      Reset
                    </button>
                    {run.status !== 'idle' && (
                      <span className="mono tiny" style={{ color: run.status === 'done' ? 'var(--accent)' : 'var(--muted)' }}>
                        {run.status === 'running' ? `${run.stage}…` : `${run.stage} — ${run.source}${run.serverMs ? ` · ${run.serverMs} ms` : ''}`}
                      </span>
                    )}
                  </div>
                  <p className="tiny dim" style={{ marginTop: 10, maxWidth: 620 }}>
                    {run.status === 'done' && run.source === 'precomputed'
                      ? '브라우저에서 추론한 것이 아니라 연구 파이프라인이 미리 계산한 결과입니다.'
                      : run.status === 'done'
                        ? '서버가 잠금 파이프라인(M1 전처리 → B0 추론)을 실제로 실행해 메시를 생성했습니다.'
                        : active.desc}
                  </p>
                </div>

                {/* ---------------------------------------------- result */}
                <div>
                  <Lab>Result</Lab>
                  <div style={{ marginTop: 14 }}>
                    <DL
                      rows={[
                        { k: 'Symmetric surface error', v: `${fmt(m.sym, 3)} mm` },
                        { k: 'P95', v: `${fmt(m.p95, 3)} mm` },
                        { k: 'Cov5', v: `${fmt(m.cov5, 2)} %` },
                        { k: 'Volume error', v: `${fmt(m.vol_err_pct, 2)} %` },
                        { k: 'Latent error (α)', v: `${fmt(m.alpha_rmse_sigma, 3)} σ` },
                        { gap: true },
                        { k: 'Pose rotation', v: `${fmt(c.metrics.clean_m1.rot_deg, 3)} °` },
                        { k: 'Pose translation', v: `${fmt(c.metrics.clean_m1.trans_mm, 3)} mm` },
                        { gap: true },
                        { k: 'Baseline (E0)', v: `${fmt(c.metrics.baseline_e0.sym, 3)} mm` },
                        { k: 'Improvement (M1 − E0)', v: <span className={dE0 < 0 ? 'acc' : 'warn'}>{signed(dE0, 3)} mm</span> },
                        { k: 'Cohort validation', v: `${research.clean.n_better} / 6 improved` },
                      ]}
                    />
                  </div>

                  <div style={{ marginTop: 34 }}>
                    <Lab>Condition comparison</Lab>
                    <table className="t" style={{ marginTop: 12 }}>
                      <thead><tr><th>condition</th><th>Sym</th><th>p95</th></tr></thead>
                      <tbody>
                        <tr className={mode === 'clean' ? 'mark' : undefined}><td className="tx">M1 normal</td><td>{fmt(c.metrics.clean_m1.sym)}</td><td>{fmt(c.metrics.clean_m1.p95)}</td></tr>
                        <tr className={mode === 'missing' ? 'mark' : undefined}><td className="tx">Main, GT+KC missing</td><td>{fmt(c.metrics.missing_main.sym)}</td><td>{fmt(c.metrics.missing_main.p95)}</td></tr>
                        <tr className={mode === 'fallback' ? 'mark' : undefined}><td className="tx">F1 fallback</td><td>{fmt(c.metrics.fallback_f1.sym)}</td><td>{fmt(c.metrics.fallback_f1.p95)}</td></tr>
                        <tr><td className="tx">E0 baseline</td><td>{fmt(c.metrics.baseline_e0.sym)}</td><td>{fmt(c.metrics.baseline_e0.p95)}</td></tr>
                      </tbody>
                    </table>
                    <div className="tiny dim" style={{ marginTop: 12 }}>
                      단위 mm · 출처 STEP37 / STEP26 · 화면 정합은 평가와 동일한 E0 pose 사용
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
