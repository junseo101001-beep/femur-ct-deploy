import { Link } from 'react-router-dom'
import { Gate } from '../data/store'
import { DEMO_BASE, caseLabel, row } from '../data/research'
import { BONE, FemurViewer } from '../components/FemurViewer'
import { Lab, Meta, Strip, fmt, t } from '../components/ui'

export default function Home() {
  return (
    <main className="page">
      <Gate>
        {({ research, manifest }) => {
          const c0 = manifest.cases[0]
          const n50m0 = row(research, 'main_noise', 'Noise_SNR50', 'M0')
          const n50m1 = row(research, 'main_noise', 'Noise_SNR50', 'M1')
          return (
            <div className="wrap">
              <div className="hero">
                <div>
                  <Lab>대퇴골 3차원 복원 / 2026</Lab>
                  <h1 style={{ marginTop: 18 }}>
                    다방향 X-ray 영상에서
                    <br />
                    3차원 대퇴골
                    <span className="l2">복원</span>
                  </h1>
                  <p className="hero-desc small" style={{ marginTop: 12 }}>
                    0°·45°·90° 세 방향의 관측만으로 개인별 대퇴골의 자세와 형태를 추정한다.
                    평가는 CT에서 생성한 DRR과 독립 검증 6명을 기준으로 한다.
                  </p>

                  <Meta
                    items={[
                      { k: 'Model', v: research.lock.main.name },
                      { k: 'SSM', v: 'N=27' },
                      { k: 'Latent', v: `K=${research.lock.main.b0_K}` },
                      { k: 'Validation', v: `n=${research.validation_cohort.length}` },
                    ]}
                  />

                  <div className="row" style={{ marginTop: 34, gap: 8 }}>
                    <Link className="tbtn wide" to="/reconstruction">복원 →</Link>
                  </div>
                </div>

                <div className="viewer-bare" style={{ height: 'min(70vh, 640px)' }}>
                  <FemurViewer
                    className="viewer-bare"
                    height="100%"
                    autoRotate
                    background="#1f2a44"
                    layers={[{ url: `${DEMO_BASE}/${c0.meshes.recon_clean.file}`, color: BONE, opacity: 1, visible: true }]}
                  >
                    <div className="viewer-note-r">
                      <div className="lab">M1 / final lock</div>
                    </div>
                    <div className="viewer-note">
                      <div className="mono tiny dim">{caseLabel(c0.pid, 0)} · {c0.meshes.recon_clean.triangles.toLocaleString()} tri · drag to rotate</div>
                    </div>
                  </FemurViewer>
                </div>
              </div>

              <Strip
                items={[
                  { k: t('Mean surface error'), v: fmt(research.clean.b0.sym, 4), u: 'mm', note: 'clean · independent validation n=6' },
                  { k: t('P95'), v: fmt(research.clean.b0.p95, 3), u: 'mm', note: 'hausdorff 95th percentile' },
                  { k: t('Cov5'), v: fmt(research.clean.b0.cov5, 2), u: '%', note: 'GT surface within 5 mm' },
                  { k: 'Improved', v: `${research.clean.n_better} / 6`, note: `vs E0 baseline ${fmt(research.clean.e0.sym, 4)} mm` },
                  {
                    k: 'Noise SNR 50',
                    v: <><span className="dim">{fmt(n50m0?.sym ?? null)}</span> → {fmt(n50m1?.sym ?? null)}</>,
                    u: 'mm', note: 'M0 → M1 (adaptive threshold)',
                  },
                ]}
              />

              <div className="g2" style={{ marginTop: 64 }}>
                <div>
                  <Lab>Scope</Lab>
                  <p className="muted small" style={{ marginTop: 12 }}>
                    입력은 모두 CT에서 생성한 DRR이며, 실제 X-ray 촬영 영상으로는 검증되지 않았다.
                    독립 검증 대상은 6명이고, 2D 기준점은 영상에서 검출한 값이 아니라 3D 좌표의 투영값이다.
                  </p>
                </div>
                <div>
                  <Lab>Final configuration</Lab>
                  <div style={{ marginTop: 12 }}>
                    <table className="t">
                      <tbody>
                        <tr><td className="tx">Main</td><td>M1 = B0 + adaptive contour threshold</td></tr>
                        <tr><td className="tx">Threshold rule</td><td className="tiny">max(0.02, corner-median + 3·1.4826·MAD)</td></tr>
                        <tr><td className="tx">Fallback</td><td>F1 · K={research.lock.fallback.K} · γ={research.lock.fallback.gamma}</td></tr>
                        <tr><td className="tx">Model md5</td><td className="tiny">{String(research.lock.main.b0_model_md5).slice(0, 16)}…</td></tr>
                        <tr><td className="tx">Locked</td><td>{research.lock.locked_at}</td></tr>
                      </tbody>
                    </table>
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
