import { useEffect, useState, type ReactNode } from 'react'
import { Gate } from '../data/store'
import { DEMO_BASE, row, type Manifest, type Research } from '../data/research'
import { Lab, PageHead, fmt } from '../components/ui'

/* ==================================================================
 * 한계를 "왜 안 됐는지" 실제 결과값으로 보여준다.
 * 모든 수치는 research.json · manifest.json · process/<pid>/trace.json 에서 읽는다.
 * 새로 계산하는 것은 표시용 축 범위와 6명 단순 평균뿐이다.
 * ================================================================== */

const C = { text: '#172235', muted: '#65717b', dim: '#9aa3aa', line: '#d9dee2', soft: '#e8ebed', acc: '#167a63', bad: '#b96d78', warn: '#b88232', grey: '#4a5257' }
const W = 560, H = 240, PAD = { l: 58, r: 22, t: 22, b: 40 }

function scale(d0: number, d1: number, r0: number, r1: number) {
  return (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
}
function YAxis({ y, lo, hi, ticks = 4, unit }: { y: (v: number) => number; lo: number; hi: number; ticks?: number; unit: string }) {
  const vs = Array.from({ length: ticks + 1 }, (_, i) => lo + ((hi - lo) * i) / ticks)
  return (
    <g className="ax">
      {vs.map((v) => (
        <g key={v}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke={C.soft} />
          <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end">{v.toFixed(Math.abs(hi - lo) < 1 ? 2 : 1)}</text>
        </g>
      ))}
      <text x={PAD.l - 8} y={PAD.t - 8} textAnchor="end" className="unit">{unit}</text>
    </g>
  )
}
function Svg({ children, label }: { children: ReactNode; label: string }) {
  return <svg className="lim-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>{children}</svg>
}

/* ------------------------------------------------------------------ 01 n = 6 */
function ChartSample({ research }: { research: Research }) {
  const ps = research.validation_cohort.map((p) => ({ p, v: research.clean.per_subject[p].b0 })).sort((q, r) => q.v - r.v)
  const lo = 1.1, hi = 1.8
  const x = scale(lo, hi, 150, W - 40)
  const mean = research.clean.b0.sym
  const y0 = 50, dy = 26
  return (
    <Svg label="validation subjects">
      <text x={20} y={26} className="cap">독립 검증 대상별 Sym (M1, 정상 입력) · n = {ps.length}</text>
      {[1.1, 1.3, 1.5, 1.7].map((t) => (
        <g key={t} className="ax">
          <line x1={x(t)} x2={x(t)} y1={y0 - 10} y2={y0 + dy * (ps.length - 1) + 10} stroke={C.soft} />
          <text x={x(t)} y={y0 + dy * (ps.length - 1) + 28} textAnchor="middle">{t.toFixed(1)}</text>
        </g>
      ))}
      <line x1={x(mean)} x2={x(mean)} y1={y0 - 16} y2={y0 + dy * (ps.length - 1) + 12} stroke={C.text} strokeDasharray="4 3" />
      <text x={x(mean) + 6} y={y0 - 20} className="lbl strong">mean {mean.toFixed(4)} mm</text>
      {ps.map((s, i) => (
        <g key={s.p}>
          <text x={140} y={y0 + i * dy + 4} textAnchor="end" className="lbl strong">{s.p}</text>
          <line x1={150} x2={x(s.v)} y1={y0 + i * dy} y2={y0 + i * dy} stroke={C.soft} />
          <circle cx={x(s.v)} cy={y0 + i * dy} r={6} fill={C.acc} />
          <text x={x(s.v) + 12} y={y0 + i * dy + 4} className="lbl">{s.v.toFixed(3)}</text>
        </g>
      ))}
      <text x={20} y={232} className="cap dim">학습 N = 27 · 독립 검증 n = 6 · 외부 데이터 0</text>
    </Svg>
  )
}

/* ------------------------------------------------------------------ 02 real X-ray 0 */
function ChartRealXray({ manifest }: { manifest: Manifest }) {
  const n = manifest.cases.length
  return (
    <Svg label="input data source">
      <text x={PAD.l} y={34} className="cap">검증에 사용된 입력 영상</text>
      {[['CT → DRR (합성 X-ray)', n, C.acc], ['실제 촬영 X-ray', 0, C.bad]].map(([t, v, col], i) => {
        const y = 70 + i * 74
        return (
          <g key={String(t)}>
            <text x={PAD.l} y={y - 8} className="lbl strong">{t}</text>
            {Array.from({ length: n }, (_, k) => (
              <rect key={k} x={PAD.l + k * 62} y={y} width={52} height={30}
                    fill={k < (v as number) ? (col as string) : 'none'} fillOpacity={0.85}
                    stroke={k < (v as number) ? 'none' : C.dim} strokeDasharray="4 3" />
            ))}
            <text x={PAD.l + n * 62 + 10} y={y + 21} className="num" fill={col as string}>{v as number} 명</text>
          </g>
        )
      })}
      <text x={PAD.l} y={226} className="cap dim">외부 데이터: HFValid 적격 0명 · VSDFullBody landmark 정의 불일치로 부적격</text>
    </Svg>
  )
}

/* ------------------------------------------------------------------ 03 projected landmarks */
interface LmView { angle_deg: number; contour: { px: [number, number][] }; landmarks: Record<string, { px: [number, number] }> }
function ChartLandmark({ manifest }: { manifest: Manifest }) {
  const c = manifest.cases[0]
  const [v, setV] = useState<LmView | null>(null)
  useEffect(() => {
    let alive = true
    fetch(`${DEMO_BASE}/process/${c.pid}/trace.json`).then((r) => r.json()).then((j) => alive && setV(j.views[0])).catch(() => {})
    return () => { alive = false }
  }, [c.pid])
  const col: Record<string, string> = { femurHead: C.acc, greatTroch: C.warn, kneeCenter: '#5b7896' }
  const ab: Record<string, string> = { femurHead: 'FH', greatTroch: 'GT', kneeCenter: 'KC' }
  return (
    <svg className="lim-svg" viewBox="0 0 560 240" role="img" aria-label="projected landmarks">
      <rect x={20} y={10} width={165} height={220} fill="#050607" />
      <image href={`${DEMO_BASE}/${c.drr['000'].file}`} x={20} y={10} width={165} height={220} preserveAspectRatio="xMidYMid meet" />
      {v && (() => {
        const sx = 165 / 450, sy = 220 / 600
        return (
          <g transform={`translate(20 10) scale(${sx} ${sy})`}>
            <polyline points={v.contour.px.map((p) => p.join(',')).join(' ')} fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth={3} />
            {Object.entries(v.landmarks).map(([k, l]) => (
              <g key={k}>
                <circle cx={l.px[0]} cy={l.px[1]} r={14} fill={col[k]} stroke="#fff" strokeWidth={4} />
                <text x={l.px[0] + 22} y={l.px[1] + 9} fill="#fff" fontSize={30} fontFamily="IBM Plex Mono">{ab[k]}</text>
              </g>
            ))}
          </g>
        )
      })()}
      <g className="flowtxt">
        <text x={215} y={46} className="lbl strong">3D ground-truth landmark</text>
        <text x={215} y={66} className="lbl">CT 표면 위 전문가 지정 좌표</text>
        <text x={228} y={96} className="arrow">↓ detector 로 투영</text>
        <text x={215} y={126} className="lbl strong">2D landmark (그림의 점)</text>
        <text x={215} y={146} className="lbl">{c.pid} · 0° · 오차 0 인 이상적 위치</text>
        <rect x={215} y={168} width={320} height={44} fill="#fbf6ec" stroke="#e3cfa9" />
        <text x={227} y={187} className="lbl" fill={C.warn}>영상에서 자동 검출한 좌표가 아님</text>
        <text x={227} y={204} className="lbl dim">실제 검출 오차는 평가에 포함되지 않음</text>
      </g>
    </svg>
  )
}

/* ------------------------------------------------------------------ 04 angle */
function ChartAngle({ research }: { research: Research }) {
  const clean = row(research, 'main_clean', 'Clean', 'M1')!.sym!
  const pts: [number, number][] = [
    [-5, row(research, 'main_angle', 'Angle_minus5', 'M1')!.sym!], [-2, row(research, 'main_angle', 'Angle_minus2', 'M1')!.sym!],
    [-1, row(research, 'main_angle', 'Angle_minus1', 'M1')!.sym!], [0, clean],
    [1, row(research, 'main_angle', 'Angle_plus1', 'M1')!.sym!], [2, row(research, 'main_angle', 'Angle_plus2', 'M1')!.sym!],
    [5, row(research, 'main_angle', 'Angle_plus5', 'M1')!.sym!],
  ]
  const [lo, hi] = [1.3, 2.0]
  const x = scale(-5.5, 5.5, PAD.l, W - PAD.r), y = scale(lo, hi, H - PAD.b, PAD.t)
  return (
    <Svg label="angle error">
      <YAxis y={y} lo={lo} hi={hi} unit="Sym mm" />
      <rect x={x(-1.5)} y={PAD.t} width={x(1.5) - x(-1.5)} height={H - PAD.b - PAD.t} fill={C.acc} fillOpacity={0.06} />
      <text x={x(0)} y={PAD.t + 14} textAnchor="middle" className="lbl" fill={C.acc}>±1° 안정</text>
      <line x1={PAD.l} x2={W - PAD.r} y1={y(clean)} y2={y(clean)} stroke={C.muted} strokeDasharray="4 3" />
      <text x={W - PAD.r} y={y(clean) - 6} textAnchor="end" className="lbl">clean {clean.toFixed(3)}</text>
      <polyline points={pts.map(([a, v]) => `${x(a)},${y(v)}`).join(' ')} fill="none" stroke={C.grey} strokeWidth={1.5} />
      {pts.map(([a, v]) => (
        <g key={a}>
          <circle cx={x(a)} cy={y(v)} r={5} fill={Math.abs(a) >= 5 ? C.bad : Math.abs(a) >= 2 ? C.warn : C.acc} />
          <text x={x(a)} y={y(v) - 10} textAnchor="middle" className="lbl">{v.toFixed(3)}</text>
          <text x={x(a)} y={H - PAD.b + 18} textAnchor="middle" className="lbl">{a > 0 ? `+${a}` : a}°</text>
        </g>
      ))}
    </Svg>
  )
}

/* ------------------------------------------------------------------ slope chart helper */
function Slope({ left, right, rows, unit, lo, hi, highlight, label }: {
  left: string; right: string; unit: string; lo: number; hi: number; highlight?: string; label: string
  rows: { p: string; a: number; b: number }[]
}) {
  const xL = PAD.l + 60, xR = W - PAD.r - 110
  const y = scale(lo, hi, H - PAD.b, PAD.t)
  let LY: number[] = []
  return (
    <Svg label={label}>
      <YAxis y={y} lo={lo} hi={hi} unit={unit} />
      <text x={xL} y={H - PAD.b + 22} textAnchor="middle" className="lbl strong">{left}</text>
      <text x={xR} y={H - PAD.b + 22} textAnchor="middle" className="lbl strong">{right}</text>
      {(() => {
        // 라벨 y 위치만 최소 14px 간격으로 벌린다 (점과 선은 실제 값 위치 그대로)
        const order = rows.map((r, i) => ({ i, y: y(r.b) })).sort((p, q) => p.y - q.y)
        const ly: number[] = []
        order.forEach((o, k) => { ly[o.i] = k === 0 ? o.y : Math.max(o.y, ly[order[k - 1].i] + 14) })
        LY = ly
        return null
      })()}
      {rows.map((r, ri) => {
        const worse = r.b > r.a, hl = highlight ? r.p === highlight : worse
        const col = worse ? C.bad : C.acc
        return (
          <g key={r.p} opacity={highlight && !hl ? 0.45 : 1}>
            <line x1={xL} x2={xR} y1={y(r.a)} y2={y(r.b)} stroke={col} strokeWidth={hl ? 2.4 : 1.4} />
            <circle cx={xL} cy={y(r.a)} r={3.5} fill={C.grey} />
            <circle cx={xR} cy={y(r.b)} r={4} fill={col} />
            <text x={xR + 10} y={(LY[ri] ?? y(r.b)) + 4} className={`lbl${hl ? ' strong' : ''}`} fill={hl ? col : undefined}>{r.p} {r.a.toFixed(2)}→{r.b.toFixed(2)}</text>
          </g>
        )
      })}
    </Svg>
  )
}

/* ------------------------------------------------------------------ 05 fallback tail */
function ChartFallback({ research }: { research: Research }) {
  const ps = research.validation_cohort
  const S = research.missing.per_subject
  const sym = ps.map((p) => ({ p, a: S[p].control.sym, b: S[p].f1.sym }))
  const p95 = ps.map((p) => ({ p, a: S[p].control.p95, b: S[p].f1.p95 }))
  return (
    <div className="lim-pair">
      <div>
        <div className="lim-sub">평균 표면 오차 (Sym) <b className="acc">{fmt(research.missing.control.sym.mean, 4)} → {fmt(research.missing.f1.sym.mean, 4)}</b></div>
        <Slope label="fallback mean" left="Main only" right="F1" unit="mm" lo={1.2} hi={2.4} rows={sym} />
      </div>
      <div>
        <div className="lim-sub">상위 5% 오차 (p95) <b className="bad">{fmt(research.missing.control.p95.mean, 3)} → {fmt(research.missing.f1.p95.mean, 3)}</b></div>
        <Slope label="fallback p95" left="Main only" right="F1" unit="mm" lo={2.8} hi={7.8} rows={p95} />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 06 Pat095 */
function ChartHardCase({ research }: { research: Research }) {
  const ps = research.validation_cohort
  const clean = ps.map((p) => ({ p, a: research.clean.per_subject[p].e0, b: research.clean.per_subject[p].b0 }))
  const S = research.missing.per_subject
  const miss = ps.map((p) => ({ p, a: S[p].control.sym, b: S[p].f1.sym }))
  return (
    <div className="lim-pair">
      <div>
        <div className="lim-sub">정상 입력 · E0 → M1 <b>{research.clean.n_better} / 6 개선</b></div>
        <Slope label="clean per subject" left="E0" right="M1" unit="Sym mm" lo={1.15} hi={1.85} rows={clean} highlight="Pat095" />
      </div>
      <div>
        <div className="lim-sub">GT+KC 누락 · Main → F1 <b>5 / 6 개선</b></div>
        <Slope label="missing per subject" left="Main only" right="F1" unit="Sym mm" lo={1.2} hi={2.4} rows={miss} highlight="Pat095" />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 07 missing combinations */
function ChartCombos() {
  const combos: { fh: boolean; gt: boolean; kc: boolean; status: 'm1' | 'f1' | 'none' }[] = [
    { fh: true, gt: true, kc: true, status: 'm1' },
    { fh: true, gt: false, kc: false, status: 'f1' },
    { fh: true, gt: false, kc: true, status: 'none' },
    { fh: true, gt: true, kc: false, status: 'none' },
    { fh: false, gt: true, kc: true, status: 'none' },
    { fh: false, gt: false, kc: true, status: 'none' },
    { fh: false, gt: true, kc: false, status: 'none' },
    { fh: false, gt: false, kc: false, status: 'none' },
  ]
  const cw = 50, x0 = 138
  return (
    <svg className="lim-svg" viewBox="0 0 560 240" role="img" aria-label="missing combinations">
      {['FH femurHead', 'GT greatTroch', 'KC kneeCenter'].map((t, r) => (
        <text key={t} x={x0 - 12} y={52 + r * 38} textAnchor="end" className="lbl strong">{t}</text>
      ))}
      <text x={x0 - 12} y={52 + 3 * 38 + 6} textAnchor="end" className="lbl strong">검증</text>
      {combos.map((c, i) => {
        const cx = x0 + i * cw + cw / 2
        const st = c.status === 'm1' ? ['M1', C.acc] : c.status === 'f1' ? ['F1', C.warn] : ['미검증', C.dim]
        return (
          <g key={i}>
            {[c.fh, c.gt, c.kc].map((on, r) => (
              <g key={r}>
                <circle cx={cx} cy={48 + r * 38} r={11} fill={on ? C.text : 'none'} stroke={on ? 'none' : C.bad} strokeWidth={1.5} />
                {!on && <path d={`M${cx - 5} ${43 + r * 38} l10 10 M${cx + 5} ${43 + r * 38} l-10 10`} stroke={C.bad} strokeWidth={1.5} />}
              </g>
            ))}
            <rect x={cx - cw / 2 + 4} y={150} width={cw - 8} height={30} fill={c.status === 'none' ? 'none' : (st[1] as string)} fillOpacity={0.14}
                  stroke={st[1] as string} strokeDasharray={c.status === 'none' ? '4 3' : undefined} />
            <text x={cx} y={170} textAnchor="middle" className="lbl" fill={st[1] as string} fontSize={11}>{st[0]}</text>
          </g>
        )
      })}
      <text x={x0} y={214} className="cap dim">● 관측 · ⊗ 누락 — 8개 조합 중 2개만 실험으로 검증됨 · 나머지는 판정 불가</text>
    </svg>
  )
}

/* ------------------------------------------------------------------ 08 volume */
function ChartVolume({ manifest, research }: { manifest: Manifest; research: Research }) {
  const vs = manifest.cases.map((c) => ({ p: c.pid, v: c.metrics.clean_m1.vol_err_pct ?? 0 }))
  const mean = vs.reduce((s, x) => s + x.v, 0) / vs.length
  const lo = -6, hi = 4
  const x = scale(lo, hi, PAD.l, W - PAD.r)
  return (
    <Svg label="volume error">
      <text x={PAD.l} y={30} className="cap">대상별 부피 오차 (M1, 정상 입력)</text>
      {[-6, -4, -2, 0, 2, 4].map((t) => (
        <g key={t} className="ax">
          <line x1={x(t)} x2={x(t)} y1={48} y2={196} stroke={t === 0 ? C.muted : C.soft} strokeDasharray={t === 0 ? '4 3' : undefined} />
          <text x={x(t)} y={212} textAnchor="middle">{t > 0 ? `+${t}` : t}%</text>
        </g>
      ))}
      <text x={x(-3)} y={62} textAnchor="middle" className="lbl" fill={C.bad}>← 작게 복원</text>
      <text x={x(2)} y={62} textAnchor="middle" className="lbl">크게 복원 →</text>
      {vs.map((s, i) => (
        <g key={s.p}>
          <line x1={x(0)} x2={x(s.v)} y1={82 + i * 19} y2={82 + i * 19} stroke={s.v < 0 ? C.bad : C.grey} strokeWidth={2} />
          <circle cx={x(s.v)} cy={82 + i * 19} r={5} fill={s.v < 0 ? C.bad : C.grey} />
          <text x={s.v < 0 ? x(0) + 8 : x(0) - 8} y={86 + i * 19} textAnchor={s.v < 0 ? 'start' : 'end'} className="lbl">{s.p} {s.v > 0 ? '+' : ''}{s.v.toFixed(2)}</text>
        </g>
      ))}
      <text x={PAD.l} y={234} className="cap dim">6명 단순 평균 {mean > 0 ? '+' : ''}{mean.toFixed(2)}% · 평균 절대 오차 {research.clean.b0.abs_vol.toFixed(2)}% · 6명 중 {vs.filter((s) => s.v < 0).length}명 과소</text>
    </Svg>
  )
}

/* ------------------------------------------------------------------ 09 blur */
function ChartBlur({ research }: { research: Research }) {
  const rows = [
    ['clean', row(research, 'main_clean', 'Clean', 'M0')!.sym!, row(research, 'main_clean', 'Clean', 'M1')!.sym!],
    ['blur low', row(research, 'main_blur', 'Blur_low', 'M0')!.sym!, row(research, 'main_blur', 'Blur_low', 'M1')!.sym!],
    ['blur medium', row(research, 'main_blur', 'Blur_medium', 'M0')!.sym!, row(research, 'main_blur', 'Blur_medium', 'M1')!.sym!],
  ] as [string, number, number][]
  const x = scale(0, 2.2, 150, W - 90)
  return (
    <Svg label="blur">
      {rows.map(([t, m0, m1], i) => {
        const y = 34 + i * 62
        return (
          <g key={t}>
            <text x={140} y={y + 22} textAnchor="end" className="lbl strong">{t}</text>
            <rect x={150} y={y} width={x(m0) - 150} height={14} fill={C.grey} />
            <text x={x(m0) + 8} y={y + 11} className="lbl">M0 {m0.toFixed(3)}</text>
            <rect x={150} y={y + 20} width={x(m1) - 150} height={14} fill={C.acc} />
            <text x={x(m1) + 8} y={y + 31} className="lbl">M1 {m1.toFixed(3)}</text>
            {m0 === m1 && <text x={W - 20} y={y + 22} textAnchor="end" className="lbl strong" fill={C.bad}>M0 = M1</text>}
          </g>
        )
      })}
      <text x={150} y={226} className="cap dim">흐림은 배경 잡음(MAD)을 키우지 않음 → T = 0.02 그대로 → M1 입력이 M0와 동일</text>
    </Svg>
  )
}

/* ------------------------------------------------------------------ list */

interface Lim { t: string; ko: string; v: string; tone: 'warn' | 'bad'; d: string; why: ReactNode; chart: ReactNode; src: string }

export default function Limitations() {
  const [open, setOpen] = useState<number | null>(0)
  const [all, setAll] = useState(() => new URLSearchParams(window.location.search).get('open') === 'all')
  return (
    <main className="page">
      <Gate>
        {({ research, manifest }) => {
          const LIMITS: Lim[] = [
            {
              t: 'Independent validation', ko: '독립 검증 대상이 6명뿐', v: 'n = 6', tone: 'warn',
              d: '모든 최종 수치는 Pat019 / 031 / 042 / 060 / 080 / 095 여섯 명에서만 나온 것이다. 신뢰구간이 넓고 새로운 집단에서의 성능은 알 수 없다.',
              why: <>검증용으로 따로 떼어 둔 대상이 6명이었고, 외부 데이터 확보 시도(STEP30–31)에서 조건에 맞는 대상을 찾지 못했다. 평균선 하나가 여섯 개의 점에 기대고 있다.</>,
              chart: <ChartSample research={research} />, src: 'research.json · clean.per_subject',
            },
            {
              t: 'No direct real X-ray validation', ko: '실제 X-ray로 검증하지 않음', v: 'CT → DRR only', tone: 'bad',
              d: '입력은 전부 CT에서 만든 DRR이다. 실제 촬영 영상의 산란·후처리·기기 차이는 반영되지 않았다.',
              why: <>실제 X-ray 촬영본과 같은 사람의 3D 정답(CT)이 짝으로 있는 데이터가 없어 비교 자체를 할 수 없었다. 빈 칸은 결과가 나쁜 것이 아니라 <b>측정하지 못한 것</b>이다.</>,
              chart: <ChartRealXray manifest={manifest} />, src: 'manifest.json · cases',
            },
            {
              t: 'Landmark assumption', ko: 'landmark는 검출값이 아닌 투영값', v: 'projected, not detected', tone: 'warn',
              d: '2D 기준점은 영상에서 검출한 값이 아니라 3D 좌표의 투영값이다. 실제 검출 오차는 이 평가에 포함되어 있지 않다.',
              why: <>자동 landmark 검출기는 만들지 않았고, 평가 규약상 CT의 3D 정답 좌표를 detector 로 투영해 사용했다. 그림의 점은 오차가 0인 이상적인 위치다.</>,
              chart: <ChartLandmark manifest={manifest} />, src: 'process/Pat019/trace.json · views[0].landmarks',
            },
            {
              t: 'Projection angle error', ko: '촬영 각도 오차', v: '±2° and above', tone: 'warn',
              d: '±1°까지는 clean과 거의 같지만 ±2°부터 저하가 시작되고 ±5°에서는 뚜렷하게 열화된다.',
              why: <>모델은 0°·45°·90° 고정 투영을 전제로 윤곽(SDF)과 landmark 를 해석한다. 실제 각도가 어긋나면 같은 뼈라도 윤곽 모양과 투영 위치가 달라져 관측 벡터가 달라진다.</>,
              chart: <ChartAngle research={research} />, src: 'research.json · final_rows main_angle',
            },
            {
              t: 'Fallback tail error', ko: 'F1의 상위 오차 악화', v: 'p95 elevated', tone: 'bad',
              d: 'F1은 평균 오차를 낮추지만 상위 5% 오차는 오히려 커진다. STEP39의 개선 시도는 실패했고 더 진행하지 않았다.',
              why: <>F1은 GT·kneeCenter 12열을 관측식에서 빼고 SDF와 femurHead만 쓴다. 평균은 6명 중 5명이 좋아졌지만, p95는 {research.validation_cohort.filter((p) => research.missing.per_subject[p].f1.p95 > research.missing.per_subject[p].control.p95).length}명에서 오히려 커졌다. STEP39에서 geometry feature를 더했지만 Sym {fmt(research.missing.geom_probe_step39.sym, 4)} mm로 기존 F1({fmt(research.missing.f1.sym.mean, 4)} mm)보다 나빠 채택하지 않았다.</>,
              chart: <ChartFallback research={research} />, src: 'research.json · missing.per_subject',
            },
            {
              t: 'Subject-specific hard cases', ko: '대상별 hard case (Pat095)', v: 'Pat095', tone: 'bad',
              d: '특정 대상에서는 개선이 아니라 악화가 나타난다. 어떤 대상이 어려운지 사전에 판별할 방법이 없다.',
              why: <>Pat095는 정상 입력(E0 → M1)과 결측 입력(Main → F1) 모두에서 6명 중 유일하게 반대 방향으로 움직였다. STEP27–28에서 원인 진단은 했지만 개선 방법은 모두 실패했다.</>,
              chart: <ChartHardCase research={research} />, src: 'research.json · clean.per_subject · missing.per_subject',
            },
            {
              t: 'Unverified missing combinations', ko: '검증하지 않은 결측 조합', v: 'GT+KC only', tone: 'warn',
              d: 'GT + kneeCenter 동시 결측 외의 결측 패턴은 검증 대상이 아니었다. 그 경우 동작은 판정 불가로 남는다.',
              why: <>실험은 "모두 관측(M1)"과 "GT·KC 동시 누락(F1)" 두 경우만 설계했다. 나머지 조합은 결과가 나쁜 것이 아니라 <b>돌려보지 않았다</b>.</>,
              chart: <ChartCombos />, src: 'STEP37 실험 설계',
            },
            {
              t: 'Volume representation bias', ko: '부피 오차', v: `abs ${research.clean.b0.abs_vol.toFixed(2)} %`, tone: 'warn',
              d: '복원된 뼈 부피가 대상에 따라 −4.15%…+1.87%로 흩어지고, 6명 중 4명은 작게 복원된다. 보정 시도는 실패했다.',
              why: <>과소 복원이 많지만 대상에 따라 부호가 섞여 있다 (그래프). STEP27–28에서 부피 편향을 분석하고 보정을 시도했지만 개선되지 않아 채택하지 않았다.</>,
              chart: <ChartVolume manifest={manifest} research={research} />, src: 'manifest.json · metrics.clean_m1.vol_err_pct',
            },
            {
              t: 'Blur degradation', ko: '흐림 조건 열화', v: 'unresolved', tone: 'bad',
              d: '흐림 조건에서는 M1의 적응형 문턱값이 작동하지 않아 열화가 그대로 남는다.',
              why: <>M1은 영상 귀퉁이 배경의 잡음 크기(MAD)를 재서 임계값을 올린다. 흐림은 뼈 경계를 번지게 할 뿐 배경 잡음을 늘리지 않아 T가 0.02로 유지되고, 입력이 M0와 완전히 같아진다.</>,
              chart: <ChartBlur research={research} />, src: 'research.json · final_rows main_blur',
            },
          ]
          const isOpen = (i: number) => all || open === i
          return (
            <div className="wrap">
              <PageHead
                label="Known limitations"
                title="해결하지 못한 것"
                note="성능 수치보다 중요한 항목이다. 각 항목을 펼치면 실제 실험 결과로 왜 해결되지 않았는지 보여준다. 그래프의 모든 값은 연구 결과 파일에서 읽는다."
              />

              <div className="lim-tools">
                <span className="lim-count">{LIMITS.length} limitations</span>
                <button className="rc-opt" onClick={() => { setAll(!all); setOpen(null) }}>{all ? '모두 접기' : '모두 펼치기'}</button>
              </div>

              <div className="lim-list">
                {LIMITS.map((l, i) => (
                  <section key={l.t} className={`lim${isOpen(i) ? ' open' : ''}`}>
                    <button className="lim-row" onClick={() => { setAll(false); setOpen(open === i ? null : i) }} aria-expanded={isOpen(i)}>
                      <span className="lim-n">{String(i + 1).padStart(2, '0')}</span>
                      <span className="lim-t">
                        <span className="en">{l.t}</span>
                        <span className="ko">{l.ko}</span>
                      </span>
                      <span className="lim-mini" aria-hidden>{l.chart}</span>
                      <span className={`lim-v ${l.tone}`}>{l.v}</span>
                      <svg className="lim-chev" width="12" height="12" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.3"><path d="M2 3.5 5 6.5l3-3" /></svg>
                    </button>
                    {isOpen(i) && (
                      <div className="lim-body">
                        <div className="lim-chart">{l.chart}</div>
                        <div className="lim-text">
                          <Lab>What</Lab>
                          <p>{l.d}</p>
                          <Lab>Why it was not solved</Lab>
                          <p className="lim-why">{l.why}</p>
                          <div className="lim-src">source · {l.src}</div>
                        </div>
                      </div>
                    )}
                  </section>
                ))}
              </div>

              <div style={{ marginTop: 48, paddingTop: 48, borderTop: '1px solid var(--line)' }}>
                <div className="g2" style={{ gap: 48 }}>
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
          )
        }}
      </Gate>
    </main>
  )
}
