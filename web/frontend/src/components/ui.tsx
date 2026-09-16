import type { ReactNode } from 'react'

export const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(d)
export const signed = (v: number, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`

export function Lab({ children, on }: { children: ReactNode; on?: boolean }) {
  return <div className={on ? 'lab on' : 'lab'}>{children}</div>
}

export function Delta({ v, d = 3 }: { v: number; d?: number }) {
  return <span className={v < 0 ? 'acc' : 'warn'}>{signed(v, d)}</span>
}

/**
 * UI 용어: 영어 원어 유지 + 짧은 한국어 병기. 전 페이지 동일 매핑.
 * 숫자·결과값은 건드리지 않고 표시 라벨에만 사용한다.
 */
export const TERM: Record<string, string> = {
  'Symmetric surface error': 'Symmetric surface error (대칭 표면 오차)',
  'Mean surface error': 'Mean surface error (평균 표면 오차)',
  'Absolute volume error': 'Absolute volume error (절대 부피 오차)',
  'P95': 'P95 (95백분위 표면 오차)',
  'Cov5': 'Cov5 (5 mm 이내 포함률)',
  'Volume error': 'Volume error (부피 오차)',
  'Latent error (α)': 'Latent error (잠재계수 오차, α)',
  'Pose rotation': 'Pose rotation (자세 회전 오차)',
  'Pose translation': 'Pose translation (자세 이동 오차)',
  'Baseline (E0)': 'Baseline (기존 모델, E0)',
  'E0 baseline': 'E0 baseline (기존 모델)',
  'Improvement (M1 − E0)': 'Improvement (개선량, M1 − E0)',
  'Cohort validation': 'Cohort validation (검증 대상 전체 결과)',
  'Condition comparison': 'Condition comparison (조건별 비교)',
  'Ground truth': 'Ground truth (실제 정답 모델)',
  'GT overlay': 'GT overlay (정답 중첩 표시)',
  'Overlay': 'Overlay (중첩 표시)',
  'M1 / normal': 'M1 / normal (최종 복원)',
  'M1 normal': 'M1 normal (최종 복원)',
  'Main / GT+KC missing': 'Main / GT+KC missing (기본 경로 / GT·KC 누락)',
  'Main, GT+KC missing': 'Main, GT+KC missing (기본 경로 / GT·KC 누락)',
  'Main only': 'Main only (기본 경로)',
  'Main only · GT+KC missing': 'Main only · GT+KC missing (기본 경로 / GT·KC 누락)',
  'F1 / GT+KC missing': 'F1 / GT+KC missing (대체 복원 / GT·KC 누락)',
  'F1 fallback': 'F1 fallback (대체 복원)',
  'F1': 'F1 (대체 복원)',
  'E0 Sym': 'E0 Sym (기존 모델)',
  'M1 Sym': 'M1 Sym (최종 복원)',
  'Δ Sym': 'Δ Sym (차이)',
  'E0 p95': 'E0 p95 (기존 모델)',
  'M1 p95': 'M1 p95 (최종 복원)',
  'M1 Cov5': 'M1 Cov5 (5 mm 이내 포함률)',
  'M1 vol err': 'M1 vol err (부피 오차)',
  'Main p95': 'Main p95 (기본 경로)',
  'F1 p95': 'F1 p95 (대체 복원)',
  'F1 Cov5': 'F1 Cov5 (5 mm 이내 포함률)',
  'P95 (main → F1)': 'P95 (main → F1) (95백분위 변화)',
  'Cov5 (main → F1)': 'Cov5 (main → F1) (포함률 변화)',
  'subject': 'subject (검증 대상)',
}
export function t(key: string): string {
  return TERM[key] ?? key
}

/** 페이지 머리 — 라벨 / 제목 / 설명 */
export function PageHead({ label, title, note }: { label: string; title: string; note?: ReactNode }) {
  return (
    <div className="head">
      <Lab>{label}</Lab>
      <h2>{title}</h2>
      {note && <p className="muted">{note}</p>}
    </div>
  )
}

/** 번호가 붙은 실험/단계 섹션 */
export function Sec({ num, title, note, right, children }: { num?: string; title: string; note?: ReactNode; right?: ReactNode; children?: ReactNode }) {
  return (
    <section className="sec">
      <div className="sec-head">
        {num && <div className="sec-num">{num}</div>}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="spread">
            <h3 className="sec-title">{title}</h3>
            {right}
          </div>
          {note && <div className="sec-note">{note}</div>}
        </div>
      </div>
      {children}
    </section>
  )
}

/** 좌측 기술 메타데이터 (MODEL / SSM / LATENT / VALIDATION) */
export function Meta({ items }: { items: { k: string; v: ReactNode }[] }) {
  return (
    <div className="meta">
      {items.map((i) => (
        <div className="meta-item" key={i.k}>
          <div className="lab">{i.k}</div>
          <div className="meta-v">{i.v}</div>
        </div>
      ))}
    </div>
  )
}

/** 하단 수치 스트립 (카드 아님) */
export function Strip({ items }: { items: { k: string; v: ReactNode; u?: string; note?: ReactNode }[] }) {
  return (
    <div className="strip">
      {items.map((i) => (
        <div className="strip-item" key={i.k}>
          <div className="lab">{i.k}</div>
          <div className="strip-v">
            {i.v}
            {i.u && <span className="u">{i.u}</span>}
          </div>
          {i.note && <div className="tiny dim" style={{ marginTop: 6 }}>{i.note}</div>}
        </div>
      ))}
    </div>
  )
}

/** 얇은 divider 기반 결과 목록 */
export function DL({ rows }: { rows: ({ k: ReactNode; v: ReactNode; gap?: false } | { gap: true })[] }) {
  return (
    <div className="dl">
      {rows.map((r, i) =>
        'gap' in r && r.gap ? (
          <div className="dl-gap" key={`g${i}`} />
        ) : (
          <div className="dl-row" key={i}>
            <span className="k">{(r as any).k}</span>
            <span className="v">{(r as any).v}</span>
          </div>
        ),
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ bars */

export interface BarSeries { key: string; label: string; tone?: 'a' | 'b' }
export interface BarRow { label: string; values: Record<string, number | null>; note?: ReactNode; emphasis?: boolean; tone?: 'a' | 'b' }

/** 얇은 수평 막대 — 절대 스케일, 장식 없음 */
export function Bars({
  rows, series, unit = 'mm', max, decimals = 3, reference,
}: { rows: BarRow[]; series: BarSeries[]; unit?: string; max?: number; decimals?: number; reference?: { value: number | null; label: string } }) {
  const all = rows.flatMap((r) => series.map((s) => r.values[s.key]).filter((v): v is number => typeof v === 'number'))
  const ref = typeof reference?.value === 'number' ? reference.value : null
  const top = max ?? Math.max(...all, ref ?? 0) * 1.1

  return (
    <div>
      {rows.map((r) => (
        <div key={r.label} style={{ padding: '10px 0', borderBottom: '1px solid var(--line-soft)' }}>
          {series.map((s, si) => {
            const v = r.values[s.key]
            if (typeof v !== 'number') return null
            return (
              <div className="bar-row" key={s.key}>
                <div className="bar-label" style={{ color: si === 0 ? (r.emphasis ? 'var(--text)' : 'var(--muted)') : 'transparent' }}>
                  {r.label}
                </div>
                <div className="bar-track">
                  {ref !== null && (
                    <div className="bar-ref" style={{ left: `${(ref / top) * 100}%` }} title={reference!.label} />
                  )}
                  <div className={`bar ${(r.tone ?? s.tone) === 'a' ? 'a' : 'b'}`} style={{ width: `${Math.max(0.5, (v / top) * 100)}%` }} />
                </div>
                <div className="bar-val">
                  {v.toFixed(decimals)}
                  <span className="dim" style={{ marginLeft: 4 }}>{unit}</span>
                </div>
              </div>
            )
          })}
          {r.note && <div className="tiny dim" style={{ marginLeft: 114 }}>{r.note}</div>}
        </div>
      ))}
      <div className="row" style={{ marginTop: 14, gap: 22 }}>
        {series.map((s) => (
          <span className="mono tiny muted" key={s.key}>
            <span style={{ display: 'inline-block', width: 16, height: 6, marginRight: 8, verticalAlign: 'middle', background: s.tone === 'a' ? 'var(--accent)' : '#4a5257' }} />
            {s.label}
          </span>
        ))}
        {reference && (
          <span className="mono tiny dim">
            <span style={{ display: 'inline-block', width: 16, borderTop: '1px dashed var(--line)', marginRight: 8, verticalAlign: 'middle' }} />
            {reference.label}
          </span>
        )}
      </div>
    </div>
  )
}

/** 대상별 개선/악화 한 줄 표시 */
export function SubjectRow({ items }: { items: { label: string; improved: boolean; value: string }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap' }}>
      {items.map((i) => (
        <div key={i.label} style={{ padding: '12px 22px 12px 0', marginRight: 22, borderRight: '1px solid var(--line-soft)' }}>
          <div className="lab">{i.label}</div>
          <div className="mono" style={{ marginTop: 5, fontSize: 13.5, color: i.improved ? 'var(--accent)' : 'var(--warn)' }}>
            {i.improved ? '−' : '+'}{i.value}
          </div>
        </div>
      ))}
    </div>
  )
}

/** editorial flow diagram */
export function Flow({ nodes, inputs }: { nodes: { n: string; d?: string }[]; inputs?: string[] }) {
  return (
    <div className="flow">
      {inputs && (
        <>
          <div className="flow-in">
            {inputs.map((i) => <span className="flow-chip" key={i}>{i}</span>)}
          </div>
          <div className="flow-bar" />
        </>
      )}
      {nodes.map((n, i) => (
        <div key={n.n} style={{ width: '100%' }}>
          <div className="flow-node">
            <div className="n">{n.n}</div>
            {n.d && <div className="d">{n.d}</div>}
          </div>
          {i < nodes.length - 1 && <div className="flow-bar" />}
        </div>
      ))}
    </div>
  )
}
