import { useState, type ReactNode } from 'react'
import { Gate } from '../data/store'
import { DEMO_BASE, MILESTONES, caseLabel, type CaseData, type Research } from '../data/research'
import { BONE, BONE_ALT, FemurViewer } from '../components/FemurViewer'
import { useDrrSteps } from '../components/drrSteps'
import { DL, fmt } from '../components/ui'

const drrUrl = (c: CaseData, k: string) => `${DEMO_BASE}/${c.drr[k].file}`
const meshUrl = (c: CaseData, k: string) => `${DEMO_BASE}/${c.meshes[k].file}`

function Panel({ num, title, sub, right, className = '', children }: { num?: string; title: string; sub?: string; right?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <section className={`ap ${className}`}>
      <div className="ap-head">
        {num && <span className="ap-num">{num}</span>}
        <span className="ap-title">{title}</span>
        {sub && <span className="ap-sub">{sub}</span>}
        {right && <span style={{ marginLeft: 'auto' }}>{right}</span>}
      </div>
      {children}
    </section>
  )
}

function Img({ src, alt, dark = true }: { src?: string; alt: string; dark?: boolean }) {
  return (
    <div className={dark ? 'pimg' : 'pimg light'}>
      {src ? <img src={src} alt={alt} /> : <span className="pimg-load">…</span>}
    </div>
  )
}

function Step({ n, title, desc, children }: { n: string; title: string; desc: string; children: ReactNode }) {
  return (
    <div className="pstep">
      <div className="pstep-n">{n}</div>
      <div className="pstep-t">{title}</div>
      <p className="pstep-d">{desc}</p>
      <div className="pstep-vis">{children}</div>
    </div>
  )
}

function MainPipeline({ c, idx }: { c: CaseData; idx: number }) {
  const d0 = c.drr['000']
  const s = useDrrSteps(drrUrl(c, '000'), d0.raw_min, d0.raw_max)
  return (
    <>
      <div className="pipe">
        <div className="pin">
          <div className="pin-lab">INPUT : MULTI-VIEW X-RAY</div>
          <div className="pin-row">
            {['000', '045', '090'].map((k) => (
              <figure key={k}>
                <Img src={drrUrl(c, k)} alt={`DRR ${k}`} />
                <figcaption>{Number(k)}°</figcaption>
              </figure>
            ))}
          </div>
          <div className="pin-note">{caseLabel(c.pid, idx)} · CT 기반 DRR</div>
        </div>

        <Step n="02" title="ADAPTIVE THRESHOLD" desc="영상 네 귀퉁이 배경의 median·MAD로 잡음 크기를 추정해 임계값을 자동으로 정한다.">
          <Img src={s?.mask} alt="threshold mask" />
          <div className="pstep-cap">T = {s ? s.T.toFixed(4) : '—'}</div>
        </Step>

        <Step n="03" title="CONTOUR + SDF" desc="closing → 구멍 채우기 → 최대 영역으로 윤곽선을 얻고, 부호 있는 거리장(SDF)으로 표현한다.">
          <div className="pair">
            <Img src={s?.contour} alt="contour" />
            <Img src={s?.sdf} alt="signed distance field" />
          </div>
          <div className="pstep-cap">inside − / outside +</div>
        </Step>

        <Step n="04" title="LANDMARK-WEIGHTED INFERENCE" desc="SDF 3블록과 landmark 블록을 각각 정규화해 합친 뒤 정규화 최소제곱으로 형상 계수를 추정한다.">
          <div className="pair">
            <Img src={s?.contour} alt="contour" dark={false} />
            <Img src={s?.samples} alt="sample points" dark={false} />
          </div>
          <div className="pstep-cap">SDF ×3 + landmark · schematic</div>
        </Step>

        <Step n="05" title="SSM" desc="N=27 CT로 만든 평균 형상과 주성분. 추정한 K개 계수만큼 평균 형상을 변형한다.">
          <FemurViewer className="pview" height={170} autoRotate background={null} shadow={false} fov={30}
            layers={[{ url: meshUrl(c, 'recon_clean'), color: '#5f9585', opacity: 0.45, visible: true, wireframe: true }]} />
          <div className="pstep-cap">4,911 vertices · wireframe</div>
        </Step>

        <Step n="06" title="3D FEMUR" desc="최종적으로 개인별 3D 대퇴골 형상을 복원한다.">
          <FemurViewer className="pview" height={170} autoRotate background={null} shadow={false} fov={30}
            layers={[{ url: meshUrl(c, 'recon_clean'), color: BONE, opacity: 1, visible: true }]} />
          <div className="pstep-cap">Sym {fmt(c.metrics.clean_m1.sym, 3)} mm</div>
        </Step>
      </div>
      <div className="ap-foot">
        02–04 이미지는 공개된 DRR PNG(8bit)에서 브라우저가 같은 규칙으로 다시 계산한 설명용 시각화다. 04의 점은 윤곽 표본 위치를 나타내는 도식이며 실제 landmark 좌표가 아니다.
      </div>
    </>
  )
}

function Glyph({ src }: { src?: string }) {
  return <span className="glyph">{src && <img src={src} alt="" />}</span>
}

function FallbackDiagram({ c, research }: { c: CaseData; research: Research }) {
  const d0 = c.drr['000']
  const s = useDrrSteps(drrUrl(c, '000'), d0.raw_min, d0.raw_max)
  const fb = research.lock.fallback
  return (
    <div className="fd">
      <div className="fd-in">
        <div className="fd-k">INPUT</div>
        <div className="fd-s">X-ray + Landmark</div>
      </div>
      <svg className="fd-fork" viewBox="0 0 40 140" preserveAspectRatio="none" aria-hidden>
        <path d="M0 70 H14 V22 H40 M14 70 V118 H40" fill="none" stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </svg>

      <div className="fd-row ok">
        <div className="fd-box ok"><div>정상 입력</div><div className="fd-s">(모든 landmark 존재)</div></div>
        <span className="fd-arr">→</span>
        <div className="fd-box ok"><div>M1</div><div className="fd-s">(Main model)</div></div>
        <span className="fd-arr">→</span>
        <div className="fd-box out"><span>3D FEMUR</span><Glyph src={s?.silhouette} /></div>
      </div>

      <div className="fd-row wa">
        <div className="fd-box wa"><div>GT + KC 누락</div><div className="fd-s">(GT · kneeCenter landmark)</div></div>
        <span className="fd-arr">→</span>
        <div className="fd-box wa">
          <span className="fd-tag">LIMITED FALLBACK</span>
          <div>F1</div><div className="fd-s">(K={fb.K}, γ={fb.gamma})</div>
        </div>
        <span className="fd-arr">→</span>
        <div className="fd-box out"><span>3D FEMUR</span><Glyph src={s?.silhouette} /></div>
      </div>
    </div>
  )
}

function Compare({ c }: { c: CaseData }) {
  const m = c.metrics
  return (
    <div className="cmp">
      <div className="cmp-card ok">
        <div className="cmp-t">M1 (정상 입력)</div>
        <div className="cmp-s">Adaptive threshold 사용</div>
        <ul>
          <li>SDF 3블록 + 전체 landmark 블록</li>
          <li>{c.pid} Sym {fmt(m.clean_m1.sym, 3)} mm · p95 {fmt(m.clean_m1.p95, 2)} mm</li>
        </ul>
        <div className="cmp-vis">
          <Img src={drrUrl(c, '000')} alt="DRR 0°" />
          <FemurViewer className="pview" height={170} autoRotate background={null} shadow={false} fov={30}
            layers={[{ url: meshUrl(c, 'recon_clean'), color: BONE, opacity: 1, visible: true }]} />
        </div>
      </div>
      <div className="cmp-card wa">
        <div className="cmp-t">F1 (GT + KC missing)</div>
        <div className="cmp-s">SDF + femurHead only</div>
        <ul>
          <li>누락 landmark 열을 관측식에서 제거</li>
          <li>{c.pid} Sym {fmt(m.missing_main.sym, 3)} → {fmt(m.fallback_f1.sym, 3)} mm · p95 {fmt(m.missing_main.p95, 2)} → {fmt(m.fallback_f1.p95, 2)} mm</li>
        </ul>
        <div className="cmp-vis">
          <Img src={drrUrl(c, '000')} alt="DRR 0°" />
          <FemurViewer className="pview" height={170} autoRotate background={null} shadow={false} fov={30}
            layers={[{ url: meshUrl(c, 'recon_fallback_f1'), color: BONE_ALT, opacity: 1, visible: true }]} />
        </div>
      </div>
      <blockquote className="cmp-q">
        <span className="cmp-qm">“</span>
        <p>F1 reduces average error in the tested missing-landmark condition, but tail error remains elevated.</p>
        <footer>FEMUR RESEARCH<br />ALGORITHM</footer>
      </blockquote>
    </div>
  )
}

export default function Algorithm() {
  const [idx, setIdx] = useState(0)
  return (
    <main className="page">
      <Gate>
        {({ research, manifest }) => {
          const c = manifest.cases[idx]
          const lock = research.lock
          return (
            <div className="wrap">
              <div className="ahead">
                <div className="ahead-num">04</div>
                <div>
                  <h1 className="ahead-t">Algorithm</h1>
                  <div className="ahead-s">From multi-view X-ray to 3D femoral shape</div>
                  <p className="ahead-p">
                    다중 뷰 X-ray(DRR) 영상을 기반으로 대퇴골의 3D 형태를 복원하는 전체 파이프라인과 모델 구성을 정리한다.
                    신경망이 형상을 직접 출력하는 방식이 아니라, 통계 형상 모델(SSM)의 계수를 2D 관측으로부터 최소제곱으로 추정한다.
                  </p>
                </div>
              </div>

              <div className="agrid-top">
                <Panel num="01" title="MAIN PIPELINE" sub="전체 파이프라인"
                  right={
                    <span className="row" style={{ gap: 4 }}>
                      {manifest.cases.map((cc, i) => (
                        <button key={cc.pid} className={`tbtn${i === idx ? ' on' : ''}`} onClick={() => setIdx(i)}>{i + 1}</button>
                      ))}
                    </span>
                  }>
                  <MainPipeline c={c} idx={idx} />
                </Panel>

                <Panel title="MODEL CONFIGURATION" sub="모델 설정" className="cfg">
                  <DL
                    rows={[
                      { k: 'Main model', v: lock.main.name },
                      { k: 'Base', v: 'B0' },
                      { k: 'Training', v: 'N = 27' },
                      { k: 'Latent', v: `K = ${lock.main.b0_K}` },
                      { k: 'Regularization', v: `γ = ${lock.main.b0_gamma}` },
                      { k: 'Views', v: '0° / 45° / 90°' },
                      { gap: true },
                      { k: 'Threshold', v: <span className="tiny">max(0.02, median + 3·1.4826·MAD)</span> },
                      { k: 'Post steps', v: <span className="tiny">closing → fill → largest</span> },
                      { k: 'Model md5', v: <span className="tiny">{String(lock.main.b0_model_md5).slice(0, 10)}…</span> },
                      { k: 'Locked', v: lock.locked_at },
                    ]}
                  />
                </Panel>
              </div>

              <div className="agrid-bot">
                <Panel num="02" title="FALLBACK PATH" sub="대체 경로">
                  <p className="ap-lead">GT + kneeCenter가 동시에 관측되지 않은 입력에서만 이 경로로 라우팅한다.</p>
                  <FallbackDiagram c={c} research={research} />
                  <div className="ap-warn">
                    <span className="ap-warn-i">!</span>
                    <span>
                      F1은 GT + kneeCenter가 누락된 경우에만 사용하는 제한적 대체 모델이며, 다른 missing 조합에 대해서는 검증되지 않았다.
                      평균 {fmt(research.missing.control.sym.mean, 4)} → {fmt(research.missing.f1.sym.mean, 4)} mm,
                      p95 {fmt(research.missing.control.p95.mean, 3)} → {fmt(research.missing.f1.p95.mean, 3)} mm (n=6).
                    </span>
                  </div>
                </Panel>

                <Panel num="03" title="KEY COMPONENTS & DIFFERENCES" sub="핵심 구성 요소 및 차이점">
                  <Compare c={c} />
                </Panel>
              </div>

              <section className="ap" style={{ marginTop: 20 }}>
                <div className="ap-head">
                  <span className="ap-num">04</span>
                  <span className="ap-title">MODEL DEVELOPMENT</span>
                  <span className="ap-sub">개발 과정 · 핵심 milestone</span>
                </div>
                <div className="tl" style={{ marginTop: 8 }}>
                  {MILESTONES.map((m) => (
                    <div className={`tl-item${m.key ? ' key' : ''}`} key={m.step + m.title}>
                      <div className="tl-step">{m.step}</div>
                      <div className="tl-title">{m.title}</div>
                      <div className="tl-note">{m.note}</div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )
        }}
      </Gate>
    </main>
  )
}
