import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Gate } from '../data/store'
import { DEMO_BASE, caseLabel, row } from '../data/research'
import { BONE, BONE_ALT, FemurViewer, GT_COLOR } from '../components/FemurViewer'
import { Bars, Lab, PageHead, Sec, fmt, t } from '../components/ui'

export default function Robustness() {
  const [showGt, setShowGt] = useState(false)
  return (
    <main className="page">
      <Gate>
        {({ research, manifest }) => {
          const r = (cat: string, cond: string, model?: string) => row(research, cat, cond, model)
          const clean = r('main_clean', 'Clean', 'M1')!
          const nd = manifest.noise_demo
          const gtFile = nd ? manifest.cases.find((c) => c.pid === nd.case_pid)!.meshes.gt.file : ''
          const m0_50 = r('main_noise', 'Noise_SNR50', 'M0')!
          const m1_50 = r('main_noise', 'Noise_SNR50', 'M1')!

          return (
            <div className="wrap">
              <PageHead
                label="Robustness / M1 under input perturbation"
                title="입력이 나빠질 때 무엇이 먼저 무너지는가"
                note="잡음·흐림·대비·촬영각·기준점 결측을 하나씩 바꿔 같은 6명에게 적용했다. M1의 유일한 변경점(적응형 문턱값)은 잡음 조건에서만 작동한다."
              />

              {/* ------------------------------------------------ 01 noise */}
              <Sec
                num="01"
                title="Gaussian noise (가우시안 잡음)"
                note="SNR이 낮아지면 기존 전처리(M0)는 윤곽 추출이 무너져 복원이 붕괴한다. M1은 배경 잡음 크기를 재서 문턱값을 올리는 것만으로 이를 막는다."
                right={<span className="mono tiny acc">4.145 → 1.469 mm @ SNR 50</span>}
              >
                <Bars
                  rows={[120, 80, 60, 50].map((s) => ({
                    label: `SNR ${s}`,
                    values: { m0: r('main_noise', `Noise_SNR${s}`, 'M0')!.sym, m1: r('main_noise', `Noise_SNR${s}`, 'M1')!.sym },
                    emphasis: s === 50,
                    note: s === 50 ? 'contour 붕괴 구간 — M1이 clean 수준으로 회복' : undefined,
                  }))}
                  series={[{ key: 'm0', label: 'M0 · fixed threshold 0.02', tone: 'b' }, { key: 'm1', label: 'M1 · adaptive threshold', tone: 'a' }]}
                  reference={{ value: clean.sym, label: `clean ${fmt(clean.sym, 3)} mm` }}
                />

                {nd && (
                  <div style={{ marginTop: 46 }}>
                    <div className="spread" style={{ marginBottom: 16 }}>
                      <Lab>Same noisy input (SNR {nd.snr}, σ {nd.sigma}) · {caseLabel(nd.case_pid, manifest.cases.findIndex((c) => c.pid === nd.case_pid))}</Lab>
                      <button className={`tbtn ${showGt ? 'on' : ''}`} onClick={() => setShowGt(!showGt)}>{t('GT overlay')}</button>
                    </div>

                    <div className="g3" style={{ gap: 14, marginBottom: 22 }}>
                      {(['000', '045', '090'] as const).map((k) => (
                        <div className="xray" key={k}>
                          <img src={`${DEMO_BASE}/${nd.drr[k].file}`} alt={`noisy DRR ${nd.drr[k].angle_deg}°`} loading="lazy" style={{ aspectRatio: '4 / 3' }} />
                          <div className="xray-cap">
                            <span className="mono tiny">{nd.drr[k].angle_deg}°</span>
                            <span className="mono tiny dim">SNR {nd.snr}</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="g2" style={{ gap: 20 }}>
                      {[
                        { key: 'noise50_m0', label: 'M0 · fixed threshold', color: BONE_ALT, v: m0_50.sym, cov: m0_50.cov5, note: '잡음 얼룩이 윤곽에 붙어 형상이 부풀고 뒤틀림' },
                        { key: 'noise50_m1', label: 'M1 · adaptive threshold', color: BONE, v: m1_50.sym, cov: m1_50.cov5, note: '같은 입력에서 clean 수준 형상 유지' },
                      ].map((x) => (
                        <div key={x.key}>
                          <FemurViewer
                            height={380}
                            background="#1f2a44"
                            layers={[
                              { url: `${DEMO_BASE}/${nd.meshes[x.key].file}`, color: x.color, opacity: showGt ? 0.5 : 1, visible: true },
                              { url: `${DEMO_BASE}/${gtFile}`, color: GT_COLOR, opacity: 0.45, visible: showGt, smooth: false },
                            ]}
                          >
                            <div className="viewer-note"><div className="mono tiny dim">{x.label}</div></div>
                          </FemurViewer>
                          <div className="spread" style={{ padding: '14px 2px 0' }}>
                            <span className="mono" style={{ fontSize: 17 }}>{fmt(x.v, 3)} <span className="tiny dim">mm</span></span>
                            <span className="mono tiny dim">{t('Cov5')} {fmt(x.cov, 2)} %</span>
                          </div>
                          <div className="tiny dim" style={{ marginTop: 6 }}>{x.note}</div>
                        </div>
                      ))}
                    </div>
                    <div className="tiny dim" style={{ marginTop: 16 }}>{nd.note}</div>
                  </div>
                )}
              </Sec>

              {/* ------------------------------------------------ 02 blur */}
              <Sec num="02" title="Blur (흐림)" note="M1의 문턱값은 잡음에만 반응하므로 흐림 조건에서는 입력이 M0와 완전히 같다. 열화가 그대로 남아 있다.">
                <Bars
                  rows={[
                    { label: 'clean', values: { v: clean.sym } },
                    { label: 'blur low', values: { v: r('main_blur', 'Blur_low', 'M1')!.sym } },
                    { label: 'blur medium', values: { v: r('main_blur', 'Blur_medium', 'M1')!.sym }, note: '열화 잔존 (M0 = M1)' },
                  ]}
                  series={[{ key: 'v', label: 'M0 = M1', tone: 'b' }]}
                  reference={{ value: clean.sym, label: 'clean' }}
                />
              </Sec>

              {/* ------------------------------------------------ 03 contrast */}
              <Sec num="03" title="Contrast (영상 대비)" note="영상 대비를 0.5배에서 1.5배까지 바꿔도 성능이 사실상 변하지 않는다.">
                <Bars
                  rows={[
                    { label: 'clean', values: { v: clean.sym } },
                    { label: '× 0.5', values: { v: r('main_contrast', 'Contrast_0.5', 'M1')!.sym } },
                    { label: '× 0.75', values: { v: r('main_contrast', 'Contrast_0.75', 'M1')!.sym } },
                    { label: '× 1.5', values: { v: r('main_contrast', 'Contrast_1.5', 'M1')!.sym } },
                  ]}
                  series={[{ key: 'v', label: 'M1', tone: 'a' }]}
                  reference={{ value: clean.sym, label: 'clean' }}
                />
              </Sec>

              {/* ------------------------------------------------ 04 angle */}
              <Sec
                num="04"
                title="Projection angle error (투영 각도 오차)"
                note="세 방향의 실제 촬영각이 가정한 각도와 다를 때다. ±1°까지는 비교적 안정적이고, ±2°부터 저하가 시작되며 ±5°에서는 뚜렷하게 열화된다."
                right={<span className="mono tiny warn">limitation</span>}
              >
                <Bars
                  rows={[
                    { label: 'clean 0°', values: { v: clean.sym } },
                    { label: '+1°', values: { v: r('main_angle', 'Angle_plus1', 'M1')!.sym } },
                    { label: '−1°', values: { v: r('main_angle', 'Angle_minus1', 'M1')!.sym } },
                    { label: '+2°', values: { v: r('main_angle', 'Angle_plus2', 'M1')!.sym } },
                    { label: '−2°', values: { v: r('main_angle', 'Angle_minus2', 'M1')!.sym } },
                    { label: '+5°', values: { v: r('main_angle', 'Angle_plus5', 'M1')!.sym }, emphasis: true },
                    { label: '−5°', values: { v: r('main_angle', 'Angle_minus5', 'M1')!.sym }, emphasis: true },
                  ]}
                  series={[{ key: 'v', label: 'M1', tone: 'b' }]}
                  reference={{ value: clean.sym, label: 'clean' }}
                />
              </Sec>

              {/* ------------------------------------------------ 05 missing */}
              <Sec
                num="05"
                title="Missing landmark (기준점 누락)"
                note="GT와 kneeCenter가 동시에 관측되지 않는 경우다. 이 조건에서만 F1 fallback으로 라우팅한다."
                right={<Link className="tbtn" to="/fallback">Fallback detail →</Link>}
              >
                <Bars
                  rows={[
                    { label: 'clean (참고)', values: { v: clean.sym } },
                    { label: t('Main only'), values: { v: research.missing.control.sym.mean } },
                    { label: t('F1 fallback'), values: { v: research.missing.f1.sym.mean }, emphasis: true, note: `평균 ${fmt(Math.abs(research.missing.delta_f1_control.mean), 4)} mm 개선 · 6명 중 5명` },
                  ]}
                  series={[{ key: 'v', label: 'mean Sym', tone: 'a' }]}
                  decimals={4}
                  reference={{ value: clean.sym, label: 'clean' }}
                />
                <div className="note-box" style={{ marginTop: 26, maxWidth: 780 }}>
                  <div className="mono tiny warn">tail error</div>
                  <p className="small muted" style={{ marginTop: 8 }}>
                    평균은 개선되지만 p95는 {fmt(research.missing.control.p95.mean, 3)} → {fmt(research.missing.f1.p95.mean, 3)} mm로 악화된다.
                    F1을 제한적 fallback으로만 채택한 이유다.
                  </p>
                </div>
              </Sec>
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
