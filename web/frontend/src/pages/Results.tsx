import { Gate } from '../data/store'
import { caseLabel } from '../data/research'
import { Bars, DL, Delta, Lab, PageHead, Sec, SubjectRow, fmt, t } from '../components/ui'

export default function Results() {
  return (
    <main className="page">
      <Gate>
        {({ research }) => {
          const { e0, b0, delta_b0_e0, n_better, per_subject } = research.clean
          const pids = research.validation_cohort
          return (
            <div className="wrap">
              <PageHead
                label="Results / independent validation"
                title="학습에 쓰이지 않은 6명에서의 정상 입력 성능"
                note="최종 모델을 한 번만 적용한 결과다. M1은 정상 입력에서 B0와 bit 단위로 동일하게 동작하므로 두 값이 같다."
              />

              <div className="g2" style={{ gap: 56 }}>
                <div>
                  <Lab>Summary</Lab>
                  <div style={{ marginTop: 16 }}>
                    <DL
                      rows={[
                        { k: t('Symmetric surface error'), v: `${fmt(b0.sym, 4)} mm` },
                        { k: t('P95'), v: `${fmt(b0.p95, 3)} mm` },
                        { k: t('Cov5'), v: `${fmt(b0.cov5, 2)} %` },
                        { k: t('Absolute volume error'), v: `${fmt(b0.abs_vol, 2)} %` },
                        { k: t('Latent error (α)'), v: `${fmt(b0.alpha, 3)} σ` },
                        { gap: true },
                        { k: t('Baseline (E0)'), v: `${fmt(e0.sym, 4)} mm` },
                        { k: 'Improvement', v: <span className="acc">{fmt(delta_b0_e0, 4)} mm</span> },
                        { k: 'Validation', v: `${n_better} / 6 improved` },
                      ]}
                    />
                  </div>
                </div>

                <div>
                  <Lab>E0 baseline vs final main (mean symmetric surface error)</Lab>
                  <div style={{ marginTop: 20 }}>
                    <Bars
                      rows={[
                        { label: 'E0 baseline', values: { v: e0.sym }, tone: 'b' as const },
                        { label: 'M1 (final)', values: { v: b0.sym }, emphasis: true, note: `${fmt(Math.abs(delta_b0_e0), 4)} mm 개선 · 6명 중 ${n_better}명` },
                      ]}
                      series={[{ key: 'v', label: 'mean Sym · absolute scale', tone: 'a' }]}
                      decimals={4}
                    />
                  </div>
                </div>
              </div>

              <Sec
                num="01"
                title="대상별 결과"
                note={`6명 중 ${n_better}명에서 E0보다 좋아졌고, Pat095에서는 E0가 더 좋았다. 평균이 좋아졌다고 모든 대상이 좋아지는 것은 아니다.`}
              >
                <SubjectRow
                  items={pids.map((p, i) => ({ label: caseLabel(p, i), improved: per_subject[p].improved, value: `${fmt(Math.abs(per_subject[p].delta), 3)} mm` }))}
                />

                <div className="tscroll" style={{ marginTop: 30 }}>
                  <table className="t">
                    <thead>
                        <tr>
                          <th>{t('subject')}</th><th>{t('E0 Sym')}</th><th>{t('M1 Sym')}</th><th>{t('Δ Sym')}</th>
                          <th>{t('E0 p95')}</th><th>{t('M1 p95')}</th><th>{t('M1 Cov5')}</th><th>{t('M1 vol err')}</th>
                        </tr>
                    </thead>
                    <tbody>
                      {pids.map((p, i) => {
                        const s = per_subject[p]
                        return (
                          <tr key={p} className={s.improved ? undefined : 'mark'}>
                            <td className="tx">{caseLabel(p, i)}</td>
                            <td className="dim">{fmt(s.e0)}</td>
                            <td>{fmt(s.b0)}</td>
                            <td><Delta v={s.delta} /></td>
                            <td className="dim">{fmt(s.p95_e0)}</td>
                            <td>{fmt(s.p95_b0)}</td>
                            <td>{fmt(s.cov5_b0, 2)}</td>
                            <td>{fmt(s.vol_b0, 2)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                    <tfoot>
                      <tr>
                        <td className="tx" style={{ color: 'var(--text)' }}>mean</td>
                        <td className="dim">{fmt(e0.sym, 4)}</td>
                        <td>{fmt(b0.sym, 4)}</td>
                        <td><Delta v={delta_b0_e0} d={4} /></td>
                        <td className="dim">{fmt(e0.p95)}</td>
                        <td>{fmt(b0.p95)}</td>
                        <td>{fmt(b0.cov5, 2)}</td>
                        <td>{fmt(b0.abs_vol, 2)}</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
                <div className="tiny dim" style={{ marginTop: 14 }}>
                  단위 mm / % · 강조 행은 E0보다 나빠진 대상
                </div>
              </Sec>

              <Sec num="02" title="지표 정의">
                <div className="g2" style={{ gap: 56 }}>
                  <DL
                    rows={[
                      { k: t('Symmetric surface error'), v: '복원 ↔ 정답 표면 거리 중앙값' },
                      { k: t('P95'), v: '오차 상위 5% 지점의 거리' },
                    ]}
                  />
                  <DL
                    rows={[
                      { k: t('Cov5'), v: '정답 표면 중 5 mm 이내 비율' },
                      { k: t('Volume error'), v: '복원 부피의 상대 오차 (음수 = 과소)' },
                    ]}
                  />
                </div>
                <p className="muted small" style={{ marginTop: 22, maxWidth: 720 }}>
                  평균이 좋아도 p95가 크면 국소적으로 크게 틀린 부분이 있다는 뜻이다. 이 연구에서 fallback을 제한적으로만
                  채택한 이유도 평균이 아니라 p95 때문이다.
                </p>
              </Sec>
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
