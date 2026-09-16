import { Gate } from '../data/store'
import { MILESTONES } from '../data/research'
import { DL, Flow, Lab, PageHead, Sec, fmt } from '../components/ui'

const MAIN_NODES = [
  { n: 'ADAPTIVE THRESHOLD', d: '영상 네 귀퉁이(배경)의 중앙값과 MAD로 잡음 크기를 추정해 문턱값을 정한다. 잡음이 없으면 기존 고정값 0.02를 그대로 쓴다.' },
  { n: 'CONTOUR', d: '문턱값 위 영역 → 3×3 closing → 구멍 채우기 → 가장 큰 덩어리만 남겨 뼈 외곽선을 얻는다.' },
  { n: 'SDF', d: '정규화한 2D 좌표계에서 윤곽선까지의 부호 있는 거리장. 안쪽이 음수, 바깥쪽이 양수.' },
  { n: 'LANDMARK-WEIGHTED INFERENCE', d: 'SDF 3블록과 landmark 블록을 각각 정규화해 합친 뒤 정규화 최소제곱으로 형상 계수를 추정한다.' },
  { n: 'SSM', d: 'N27명의 CT로 만든 평균 형상과 주성분. 추정한 계수만큼 평균 형상을 변형한다.' },
  { n: '3D FEMUR', d: '4,911 정점의 대응 메시. 평가 시 E0 pose로 촬영 좌표계에 배치해 GT와 비교한다.' },
]

const FALLBACK_NODES = [
  { n: 'SDF + FEMURHEAD ONLY', d: '없는 기준점을 평균값으로 채우는 대신 관측식에서 해당 열을 제거한다.' },
  { n: 'F1 (K=20, γ=0.01)', d: '잠금된 B0의 형상 prior를 그대로 쓰고 관측 구성만 바꾼 제한적 fallback.' },
  { n: '3D FEMUR', d: '평균 오차는 줄지만 꼬리 오차(p95)는 악화가 남는다.' },
]

export default function Algorithm() {
  return (
    <main className="page">
      <Gate>
        {({ research }) => (
          <div className="wrap">
            <PageHead
              label="Algorithm / reconstruction pipeline"
              title="복원은 어떤 순서로 이루어지는가"
              note="학습된 신경망이 3차원 형상을 직접 출력하는 방식이 아니라, 통계 형상 모델(SSM)의 계수를 2D 관측으로부터 최소제곱으로 추정하는 방식이다."
            />

            <Sec num="01" title="Main path — M1">
              <div className="g2" style={{ gap: 64, alignItems: 'start' }}>
                <Flow nodes={MAIN_NODES} inputs={['X-RAY 01 / 0°', 'X-RAY 02 / 45°', 'X-RAY 03 / 90°']} />
                <div>
                  <Lab>Configuration</Lab>
                  <div style={{ marginTop: 14 }}>
                    <DL
                      rows={[
                        { k: 'Main model', v: research.lock.main.name },
                        { k: 'Base', v: 'B0' },
                        { k: 'Training', v: 'N = 27' },
                        { k: 'Latent', v: `K = ${research.lock.main.b0_K}` },
                        { k: 'Regularization', v: `γ = ${research.lock.main.b0_gamma}` },
                        { k: 'Views', v: '0° / 45° / 90°' },
                        { gap: true },
                        { k: 'Threshold', v: <span className="tiny">max(0.02, median + 3·1.4826·MAD)</span> },
                        { k: 'Post steps', v: <span className="tiny">closing → fill → largest</span> },
                        { k: 'Model md5', v: <span className="tiny">{String(research.lock.main.b0_model_md5).slice(0, 16)}…</span> },
                        { k: 'Locked', v: research.lock.locked_at },
                      ]}
                    />
                  </div>
                </div>
              </div>
            </Sec>

            <Sec num="02" title="Fallback path — F1" note="GT + kneeCenter가 동시에 관측되지 않은 입력에서만 이 경로로 라우팅한다.">
              <div className="g2" style={{ gap: 64, alignItems: 'start' }}>
                <Flow nodes={FALLBACK_NODES} inputs={['GT + KC MISSING']} />
                <div>
                  <div className="note-box">
                    <div className="mono tiny warn">F1 is a limited fallback, not the main reconstruction model.</div>
                    <p className="small muted" style={{ marginTop: 10 }}>
                      평균 오차 {fmt(research.missing.control.sym.mean, 4)} → {fmt(research.missing.f1.sym.mean, 4)} mm,
                      p95 {fmt(research.missing.control.p95.mean, 3)} → {fmt(research.missing.f1.p95.mean, 3)} mm.
                    </p>
                  </div>
                  <div style={{ marginTop: 26 }}>
                    <Lab>Routing</Lab>
                    <table className="t" style={{ marginTop: 12 }}>
                      <tbody>
                        {research.lock.routing.map((rt: { case: string; use: string }) => (
                          <tr key={rt.case}>
                            <td className="tx" style={{ maxWidth: 320 }}>{rt.case}</td>
                            <td style={{ color: rt.use.includes('검증') ? 'var(--warn)' : 'var(--text)' }}>{rt.use}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </Sec>

            <Sec num="03" title="Model development" note="채택된 것만이 아니라 무엇이 왜 실패했는지가 최종 한계를 정의한다. 핵심 milestone만 표시한다.">
              <div className="tl">
                {MILESTONES.map((m) => (
                  <div className={`tl-item${m.key ? ' key' : ''}`} key={m.step + m.title}>
                    <div className="tl-step">{m.step}</div>
                    <div className="tl-title">{m.title}</div>
                    <div className="tl-note">{m.note}</div>
                  </div>
                ))}
              </div>
              <div className="tiny dim" style={{ marginTop: 10 }}>
                source of truth: {research.source_of_truth.join(' · ')}
              </div>
            </Sec>
          </div>
        )}
      </Gate>
    </main>
  )
}
