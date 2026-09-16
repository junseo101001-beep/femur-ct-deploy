import { useState } from 'react'
import { Gate } from '../data/store'
import { DEMO_BASE, caseLabel } from '../data/research'
import { BONE, BONE_ALT, FemurViewer, GT_COLOR } from '../components/FemurViewer'
import { DL, Delta, Flow, Lab, PageHead, Sec, fmt, t } from '../components/ui'

export default function Fallback() {
  const [caseIdx, setCaseIdx] = useState(4)   // Pat080 — p95 한계가 가장 잘 드러나는 대상
  const [showGt, setShowGt] = useState(false)
  return (
    <main className="page">
      <Gate>
        {({ research, manifest }) => {
          const ctrl = research.missing.control
          const f1 = research.missing.f1
          const d = research.missing.delta_f1_control
          const c = manifest.cases[caseIdx]
          const ps = research.missing.per_subject[c.pid]

          return (
            <div className="wrap">
              <PageHead
                label="Fallback / missing landmark case"
                title="GT + kneeCenter가 동시에 없을 때"
                note="두 기준점이 함께 관측되지 않는 입력에서만 사용하는 제한적 경로입니다. 정상 입력에는 적용하지 않습니다."
              />

              <div className="g2" style={{ gap: 64, alignItems: 'start' }}>
                <div>
                  <Lab>Routing</Lab>
                  <div style={{ marginTop: 18 }}>
                    <Flow nodes={[{ n: 'NORMAL INPUT → M1', d: '모든 기준점이 관측된 경우 Main 모델을 그대로 사용한다.' }]} />
                    <div style={{ height: 22 }} />
                    <Flow nodes={[{ n: 'GT + KC MISSING → F1', d: '결측 열을 평균으로 채우지 않고 관측식에서 제거한 뒤 같은 형상 prior로 추정한다.' }]} />
                  </div>
                </div>

                <div>
                  <Lab>Result (n=6, GT+KC missing)</Lab>
                  <div style={{ marginTop: 18 }}>
                    <DL
                      rows={[
                        { k: t('Main only'), v: `${fmt(ctrl.sym.mean, 4)} mm` },
                        { k: t('F1 fallback'), v: `${fmt(f1.sym.mean, 4)} mm` },
                        { k: 'Improvement', v: <Delta v={d.mean} d={4} /> },
                        { k: 'Improved subjects', v: '5 / 6' },
                        { gap: true },
                        { k: t('P95 (main → F1)'), v: <span className="warn">{fmt(ctrl.p95.mean, 3)} → {fmt(f1.p95.mean, 3)} mm</span> },
                        { k: t('Cov5 (main → F1)'), v: `${fmt(ctrl.cov5.mean, 2)} → ${fmt(f1.cov5.mean, 2)} %` },
                        { gap: true },
                        { k: 'STEP39 geometry probe', v: <span className="warn">{fmt(research.missing.geom_probe_step39.sym, 4)} mm · 미채택</span> },
                        { k: 'Adopted fallback', v: research.missing.geom_probe_step39.adopted_fallback ?? 'FALLBACK_BASE (F1)' },
                      ]}
                    />
                  </div>
                  <div className="note-box" style={{ marginTop: 24 }}>
                    <p className="small muted" style={{ margin: 0 }}>
                      F1 reduces average error in the tested missing-landmark condition, but tail error remains elevated.
                    </p>
                  </div>
                </div>
              </div>

              <Sec num="01" title="대상별 결과" note="한 명(Pat095)에서는 fallback이 오히려 나빠집니다. 어떤 대상이 어려운지 미리 판별할 방법은 아직 없습니다.">
                <div className="tscroll">
                  <table className="t">
                    <thead><tr><th>{t('subject')}</th><th>{t('Main only')}</th><th>{t('F1')}</th><th>{t('Δ Sym')}</th><th>{t('Main p95')}</th><th>{t('F1 p95')}</th><th>{t('F1 Cov5')}</th></tr></thead>
                    <tbody>
                      {research.validation_cohort.map((p, i) => {
                        const s = research.missing.per_subject[p]
                        return (
                          <tr key={p} className={s.delta > 0 ? 'mark' : undefined}>
                            <td className="tx">{caseLabel(p, i)}</td>
                            <td className="dim">{fmt(s.control.sym)}</td>
                            <td>{fmt(s.f1.sym)}</td>
                            <td><Delta v={s.delta} /></td>
                            <td className="dim">{fmt(s.control.p95)}</td>
                            <td>{fmt(s.f1.p95)}</td>
                            <td>{fmt(s.f1.cov5, 2)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                    <tfoot>
                      <tr>
                        <td className="tx" style={{ color: 'var(--text)' }}>mean</td>
                        <td className="dim">{fmt(ctrl.sym.mean, 4)}</td>
                        <td>{fmt(f1.sym.mean, 4)}</td>
                        <td><Delta v={d.mean} d={4} /></td>
                        <td className="dim">{fmt(ctrl.p95.mean, 3)}</td>
                        <td className="warn">{fmt(f1.p95.mean, 3)}</td>
                        <td>{fmt(f1.cov5.mean, 2)}</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
                <div className="tiny dim" style={{ marginTop: 14 }}>단위 mm / % · 출처 STEP37/results/validation_f1_results.json</div>
              </Sec>

              <Sec num="02" title="같은 결측 입력, 두 경로의 3D 결과">
                <div className="spread" style={{ marginBottom: 16 }}>
                  <div className="row" style={{ gap: 6 }}>
                    <span className="lab" style={{ marginRight: 8 }}>Case</span>
                    {manifest.cases.map((cc, i) => (
                      <button key={cc.id} className={`tbtn ${i === caseIdx ? 'on' : ''}`} onClick={() => setCaseIdx(i)}>{caseLabel(cc.pid, i)}</button>
                    ))}
                  </div>
                  <button className={`tbtn ${showGt ? 'on' : ''}`} onClick={() => setShowGt(!showGt)}>{t('GT overlay')}</button>
                </div>

                <div className="g2" style={{ gap: 20 }}>
                  {[
                    { key: 'recon_missing_main', label: t('Main only · GT+KC missing'), color: BONE_ALT, v: ps.control.sym, p95: ps.control.p95 },
                    { key: 'recon_fallback_f1', label: t('F1 fallback'), color: BONE, v: ps.f1.sym, p95: ps.f1.p95 },
                  ].map((x) => (
                    <div key={x.key}>
                      <FemurViewer
                        key={c.id + x.key}
                        height={400}
                        background="#12161a"
                        layers={[
                          { url: `${DEMO_BASE}/${c.meshes[x.key].file}`, color: x.color, opacity: showGt ? 0.5 : 1, visible: true },
                          { url: `${DEMO_BASE}/${c.meshes.gt.file}`, color: GT_COLOR, opacity: 0.45, visible: showGt, smooth: false },
                        ]}
                      >
                        <div className="viewer-note"><div className="mono tiny dim">{x.label} · {caseLabel(c.pid, caseIdx)}</div></div>
                      </FemurViewer>
                      <div className="spread" style={{ padding: '14px 2px 0' }}>
                        <span className="mono" style={{ fontSize: 17 }}>{fmt(x.v, 3)} <span className="tiny dim">mm</span></span>
                        <span className="mono tiny dim">p95 {fmt(x.p95, 3)} mm</span>
                      </div>
                    </div>
                  ))}
                </div>
              </Sec>
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
