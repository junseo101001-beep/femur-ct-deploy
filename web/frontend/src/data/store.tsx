import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { loadResearch, type Manifest, type Research } from './research'

interface Store { research: Research | null; manifest: Manifest | null; error: string | null; loading: boolean }

const Ctx = createContext<Store>({ research: null, manifest: null, error: null, loading: true })

export function ResearchProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState<Store>({ research: null, manifest: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    loadResearch()
      .then(({ research, manifest }) => alive && setS({ research, manifest, error: null, loading: false }))
      .catch((e: Error) => alive && setS({ research: null, manifest: null, error: e.message, loading: false }))
    return () => { alive = false }
  }, [])
  return <Ctx.Provider value={s}>{children}</Ctx.Provider>
}

export function useResearch() { return useContext(Ctx) }

/** 로딩/에러를 한 줄로 처리하는 헬퍼. 데이터가 준비되면 render(data) 를 호출한다. */
export function Gate({ children }: { children: (d: { research: Research; manifest: Manifest }) => ReactNode }) {
  const { research, manifest, error, loading } = useResearch()
  if (loading) return <div className="label" style={{ padding: '60px 0' }}>LOADING RESEARCH DATA…</div>
  if (error || !research || !manifest)
    return (
      <div className="notice" style={{ marginTop: 24 }}>
        <div className="label" style={{ color: 'var(--warn)' }}>DEMO DATA NOT FOUND</div>
        <p className="small muted" style={{ margin: '8px 0 0' }}>
          {error ?? 'unknown error'} — <span className="mono">python web/export_demo_data.py</span> 를 실행해 demo_data 를 생성한다.
        </p>
      </div>
    )
  return <>{children({ research, manifest })}</>
}
