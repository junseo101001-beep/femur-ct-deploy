import { Suspense, useEffect, useMemo, useRef, type ReactNode } from 'react'
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { ContactShadows, GizmoHelper, GizmoViewport, OrbitControls, useProgress } from '@react-three/drei'
import { STLLoader, mergeVertices } from 'three-stdlib'
import * as THREE from 'three'
import { t } from './ui'

export interface Layer {
  url: string
  color: string
  opacity: number
  visible: boolean
  wireframe?: boolean
  smooth?: boolean
  roughness?: number
}

export type Preset = 'free' | 'front' | 'side' | 'top'

const D = 4.15
const PRESET_POS: Record<Exclude<Preset, 'free'>, [number, number, number]> = {
  front: [0, 0, D],
  side: [D, 0, 0.001],
  top: [0.001, D, 0.001],
}

/** bone / ivory — 장식 없는 재질 */
export const BONE = '#e3ddd1'
export const BONE_ALT = '#9aa3a9'
export const GT_COLOR = '#c9a86a'

function useFemurGeometry(url: string, smooth = true) {
  const raw = useLoader(STLLoader, url)
  return useMemo(() => {
    const g = raw.clone()
    g.deleteAttribute('normal')
    const merged = smooth ? mergeVertices(g, 1e-5) : g
    merged.computeVertexNormals()
    merged.computeBoundingSphere()
    return merged
  }, [raw, smooth])
}

function FemurMesh({ layer }: { layer: Layer }) {
  const geom = useFemurGeometry(layer.url, layer.smooth !== false)
  const transparent = layer.opacity < 0.999
  return (
    <mesh geometry={geom} visible={layer.visible}>
      <meshStandardMaterial
        color={layer.color}
        metalness={0.02}
        roughness={layer.roughness ?? 0.66}
        transparent={transparent}
        opacity={layer.opacity}
        depthWrite={!transparent}
        wireframe={layer.wireframe}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function CameraRig({ preset, resetToken }: { preset: Preset; resetToken: number }) {
  const { camera, controls } = useThree() as any
  const target = useRef<THREE.Vector3 | null>(null)

  useEffect(() => {
    if (preset !== 'free') target.current = new THREE.Vector3(...PRESET_POS[preset])
  }, [preset])

  useEffect(() => {
    if (resetToken > 0) {
      target.current = new THREE.Vector3(...PRESET_POS.front)
      if (controls) controls.target.set(0, 0, 0)
    }
  }, [resetToken, controls])

  useFrame(() => {
    if (!target.current) return
    camera.position.lerp(target.current, 0.11)
    camera.lookAt(0, 0, 0)
    if (camera.position.distanceTo(target.current) < 0.004) target.current = null
  })
  return null
}

/** 절제된 3점 조명 — glow 없음 */
function Lights() {
  return (
    <>
      <hemisphereLight args={['#f4f1ea', '#4f4b44', 0.55]} />
      <directionalLight position={[2.6, 4.4, 4.2]} intensity={2.0} color="#fffaf2" />
      <directionalLight position={[-4.2, 0.6, -1.6]} intensity={0.34} color="#9fb0ba" />
      <directionalLight position={[0.4, -3.4, -3.6]} intensity={0.3} color="#7f8a92" />
      <ambientLight intensity={0.5} />
    </>
  )
}

function Loading() {
  const { active, progress } = useProgress()
  if (!active) return null
  return <div className="viewer-load">LOADING MESH {progress.toFixed(0)}%</div>
}

export interface ViewerProps {
  layers: Layer[]
  height?: number | string
  autoRotate?: boolean
  background?: string | null      // null = 투명(페이지 배경 그대로)
  preset?: Preset
  resetToken?: number
  shadow?: boolean
  fov?: number
  className?: string
  axes?: boolean                  // 카메라와 동기화된 축 표시
  children?: ReactNode            // overlay label
}

export function FemurViewer({
  layers, height = 460, autoRotate = true, background = null,
  preset = 'free', resetToken = 0, shadow, fov = 32, className = 'viewer', axes = false, children,
}: ViewerProps) {
  const visible = layers.filter((l) => l.visible)
  const useShadow = shadow ?? background === null   // 프레임 없는 hero 에서만 기본 on
  return (
    <div className={className} style={{ height }}>
      <Canvas
        style={{ height: '100%', width: '100%', display: 'block' }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: background === null, preserveDrawingBuffer: true }}
        camera={{ position: PRESET_POS.front, fov, near: 0.1, far: 100 }}
      >
        {background !== null && <color attach="background" args={[background]} />}
        <Lights />
        <Suspense fallback={null}>
          {visible.map((l) => (
            <FemurMesh key={l.url + l.color} layer={l} />
          ))}
          {useShadow && (
            <ContactShadows position={[0, -1.24, 0]} opacity={0.3} scale={6} blur={3.4} far={2.2} resolution={512} color="#000000" />
          )}
        </Suspense>
        <OrbitControls
          makeDefault
          enablePan
          enableDamping
          dampingFactor={0.075}
          autoRotate={autoRotate}
          autoRotateSpeed={0.62}
          minDistance={1.7}
          maxDistance={9}
          target={[0, 0, 0]}
        />
        <CameraRig preset={preset} resetToken={resetToken} />
        {axes && (
          <GizmoHelper alignment="bottom-left" margin={[70, 76]}>
            <GizmoViewport axisColors={['#b96d78', '#167a63', '#5b7896']} labelColor="#ffffff" axisHeadScale={0.8} hideNegativeAxes />
          </GizmoHelper>
        )}
      </Canvas>
      {children}
      <Loading />
    </div>
  )
}

/* ------------------------------------------------------------------ controls */

export interface ViewerState {
  autoRotate: boolean
  wireframe: boolean
  opacity: number
  gtOpacity: number
  showGt: boolean
  preset: Preset
  resetToken: number
}

export const initialViewerState: ViewerState = {
  autoRotate: true, wireframe: false, opacity: 1, gtOpacity: 0.5, showGt: false, preset: 'free', resetToken: 0,
}

export function ViewerControls({
  s, set, gt = true,
}: { s: ViewerState; set: (p: Partial<ViewerState>) => void; gt?: boolean }) {
  return (
    <div className="tools">
      <div className="tool-group">
        <div className="lab">카메라</div>
        <div className="tool-row">
          <button className="tbtn" onClick={() => set({ preset: 'free', resetToken: s.resetToken + 1 })}>초기화</button>
          {([['front', '정면'], ['side', '측면'], ['top', '위']] as const).map(([p, label]) => (
            <button key={p} className={`tbtn ${s.preset === p ? 'on' : ''}`} onClick={() => set({ preset: p })}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="tool-group">
        <div className="lab">화면</div>
        <div className="tool-row">
          <label className="check">
            <input type="checkbox" checked={s.wireframe} onChange={(e) => set({ wireframe: e.target.checked })} />
            와이어프레임
          </label>
          <label className="check" style={{ marginLeft: 16 }}>
            <input type="checkbox" checked={s.autoRotate} onChange={(e) => set({ autoRotate: e.target.checked })} />
            자동 회전
          </label>
        </div>
        <div className="tool-row">
          <span className="lab" style={{ minWidth: 52 }}>불투명도</span>
          <input type="range" min={0.15} max={1} step={0.01} value={s.opacity} onChange={(e) => set({ opacity: Number(e.target.value) })} />
          <span className="mono tiny muted" style={{ width: 34 }}>{Math.round(s.opacity * 100)}%</span>
        </div>
      </div>

      {gt && (
        <div className="tool-group">
          <div className="lab">{t('Ground truth')}</div>
          <div className="tool-row">
            <button
              className={`tbtn ${s.showGt ? 'on' : ''}`}
              onClick={() => set({ showGt: !s.showGt, opacity: s.showGt ? 1 : 0.5 })}
            >{t('Overlay')}</button>
            <input type="range" min={0.1} max={1} step={0.01} value={s.gtOpacity} disabled={!s.showGt}
                   onChange={(e) => set({ gtOpacity: Number(e.target.value) })} />
            <span className="mono tiny muted" style={{ width: 34 }}>{Math.round(s.gtOpacity * 100)}%</span>
          </div>
        </div>
      )}
    </div>
  )
}
