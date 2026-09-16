import { Gate } from '../data/store'
import { Lab, PageHead, fmt } from '../components/ui'

const LIMITS = [
  { t: 'Independent validation', v: 'n = 6', d: '모든 최종 수치는 Pat019 / 031 / 042 / 060 / 080 / 095 여섯 명에서만 나온 것이다. 신뢰구간이 넓고 새로운 집단에서의 성능은 알 수 없다.' },
  { t: 'No direct real X-ray validation', v: 'CT → DRR only', d: '입력은 전부 CT에서 만든 DRR이다. 실제 촬영 영상의 산란·후처리·기기 차이는 반영되지 않았다.' },
  { t: 'Landmark assumption', v: 'projected, not detected', d: '2D 기준점은 영상에서 검출한 값이 아니라 3D 좌표의 투영값이다. 실제 검출 오차는 이 평가에 포함되어 있지 않다.' },
  { t: 'Projection angle error', v: '±2° and above', d: '±1°까지는 비교적 안정적이지만 ±2°부터 저하가 시작되고 ±5°에서는 뚜렷하게 열화된다.' },
  { t: 'Fallback tail error', v: 'p95 elevated', d: 'F1은 평균 오차를 낮추지만 상위 5% 오차는 오히려 커진다. STEP39의 개선 시도는 실패했고 더 진행하지 않았다.' },
  { t: 'Subject-specific hard cases', v: 'Pat095', d: '특정 대상에서는 개선이 아니라 악화가 나타난다. 어떤 대상이 어려운지 사전에 판별할 방법이 없다.' },
  { t: 'Unverified missing combinations', v: 'GT+KC only', d: 'GT + kneeCenter 동시 결측 외의 결측 패턴은 검증 대상이 아니었다. 그 경우 동작은 판정 불가로 남는다.' },
  { t: 'Volume representation bias', v: 'systematic', d: '복원된 뼈 부피가 체계적으로 작게 나오는 경향이 남아 있다. 보정 시도는 실패했다.' },
  { t: 'Blur degradation', v: 'unresolved', d: '흐림 조건에서는 M1의 적응형 문턱값이 작동하지 않아 열화가 그대로 남는다.' },
]

export default function Limitations() {
  return (
    <main className="page">
      <Gate>
        {({ research }) => (
          <div className="wrap">
            <PageHead
              label="Known limitations"
              title="이 연구가 아직 하지 못한 것"
              note="성능 수치보다 중요한 항목이다. 아래는 STEP38 최종 종합에서 미해결로 기록된 내용이며 모두 실제 실험에서 확인된 것이다."
            />

            <div>
              {LIMITS.map((l, i) => (
                <div key={l.t} style={{ display: 'grid', gridTemplateColumns: '48px minmax(0, 1fr) 200px', gap: 24, padding: '22px 0', borderTop: i === 0 ? '1px solid var(--line)' : '1px solid var(--line-soft)' }}>
                  <div className="mono tiny dim">{String(i + 1).padStart(2, '0')}</div>
                  <div>
                    <div style={{ fontSize: 15 }}>{l.t}</div>
                    <p className="muted small" style={{ marginTop: 6, maxWidth: 680 }}>{l.d}</p>
                  </div>
                  <div className="mono tiny" style={{ color: 'var(--warn)', textAlign: 'right', paddingTop: 3 }}>{l.v}</div>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 56, paddingTop: 34, borderTop: '1px solid var(--line)' }}>
              <div className="g2" style={{ gap: 56 }}>
                <div>
                  <Lab>Evidence base</Lab>
                  <table className="t" style={{ marginTop: 14 }}>
                    <tbody>
                      <tr><td className="tx">Independent validation subjects</td><td>{research.validation_cohort.length}</td></tr>
                      <tr><td className="tx">Real X-ray subjects</td><td>0</td></tr>
                      <tr><td className="tx">Training subjects</td><td>27</td></tr>
                      <tr><td className="tx">Absolute volume error</td><td>{fmt(research.clean.b0.abs_vol, 2)} %</td></tr>
                      <tr><td className="tx">External dataset accepted</td><td>0 (HFValid 0명 / VSDFullBody 부적격)</td></tr>
                    </tbody>
                  </table>
                </div>
                <div>
                  <Lab>Evaluation condition</Lab>
                  <p className="muted small" style={{ marginTop: 14, maxWidth: 560 }}>
                    모든 수치는 CT에서 생성한 합성 X-ray(DRR) 환경에서 6명을 대상으로 측정된 것이다.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </Gate>
    </main>
  )
}
