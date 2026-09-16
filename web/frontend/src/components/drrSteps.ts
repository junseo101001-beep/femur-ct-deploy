import { useEffect, useState } from 'react'

/**
 * 브라우저에서 DRR PNG 한 장으로 M1 전처리 단계를 다시 그려 보여주기 위한 시각화 헬퍼.
 * 규칙은 STEP35 s35_common.background_stats / processed 와 동일한 식을 따른다
 * (corner 40px, T = max(0.02, median + 3·1.4826·MAD)). PNG 는 8bit 로 양자화되어 있으므로
 * 여기서 계산한 값은 설명용이며 연구 결과 수치를 대체하지 않는다.
 */
export interface DrrSteps {
  xray: string
  mask: string
  contour: string
  sdf: string
  samples: string
  silhouette: string
  T: number
  med: number
  sig: number
  w: number
  h: number
}

const load = (src: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const im = new Image()
    im.onload = () => res(im)
    im.onerror = rej
    im.src = src
  })

const median = (a: Float64Array) => {
  const s = Array.from(a).sort((x, y) => x - y)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function morph(src: Uint8Array, w: number, h: number, dilate: boolean) {
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = dilate ? 0 : 1
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx
          const p = yy < 0 || yy >= h || xx < 0 || xx >= w ? 0 : src[yy * w + xx]
          if (dilate && p) v = 1
          if (!dilate && !p) v = 0
        }
      out[y * w + x] = v
    }
  return out
}

function flood(mask: Uint8Array, w: number, h: number, seeds: number[], match: (i: number) => boolean) {
  const seen = new Uint8Array(w * h)
  const stack = seeds.filter((i) => match(i))
  stack.forEach((i) => (seen[i] = 1))
  const out: number[] = []
  while (stack.length) {
    const i = stack.pop()!
    out.push(i)
    const x = i % w, y = (i / w) | 0
    const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]
    for (const j of nb) if (j >= 0 && !seen[j] && match(j)) { seen[j] = 1; stack.push(j) }
  }
  void mask
  return out
}

type Box = [number, number, number, number]   // x, y, w, h

function crop(src: HTMLCanvasElement, box: Box) {
  const c = document.createElement('canvas')
  c.width = box[2]; c.height = box[3]
  c.getContext('2d')!.drawImage(src, box[0], box[1], box[2], box[3], 0, 0, box[2], box[3])
  return c.toDataURL('image/png')
}

function toUrl(w: number, h: number, paint: (d: Uint8ClampedArray) => void, box: Box) {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  const id = ctx.createImageData(w, h)
  paint(id.data)
  ctx.putImageData(id, 0, 0)
  return crop(c, box)
}

// 절제된 순차 colormap (navy → teal → pale sand)
const STOPS: [number, [number, number, number]][] = [
  [0, [28, 42, 68]], [0.45, [44, 96, 120]], [0.7, [46, 134, 112]], [0.86, [150, 190, 120]], [1, [236, 226, 180]],
]
function cmap(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t))
  for (let k = 1; k < STOPS.length; k++) {
    if (t <= STOPS[k][0]) {
      const [t0, c0] = STOPS[k - 1], [t1, c1] = STOPS[k]
      const u = (t - t0) / (t1 - t0)
      return [0, 1, 2].map((i) => c0[i] + (c1[i] - c0[i]) * u) as [number, number, number]
    }
  }
  return STOPS[STOPS.length - 1][1]
}

export async function computeSteps(url: string, rawMin: number, rawMax: number): Promise<DrrSteps> {
  const im = await load(url)
  const w = im.naturalWidth, h = im.naturalHeight
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')!
  ctx.drawImage(im, 0, 0)
  const px = ctx.getImageData(0, 0, w, h).data
  const raw = new Float64Array(w * h)
  for (let i = 0; i < w * h; i++) raw[i] = rawMin + (px[i * 4] / 255) * (rawMax - rawMin)

  // adaptive threshold — corner k=40
  const k = 40, cor: number[] = []
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if ((y < k || y >= h - k) && (x < k || x >= w - k)) cor.push(raw[y * w + x])
  const ca = Float64Array.from(cor)
  const med = median(ca)
  const sig = 1.4826 * median(ca.map((v) => Math.abs(v - med)))
  const T = Math.max(0.02, med + 3 * sig)

  // threshold → closing → fill holes → largest component
  let m = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) m[i] = raw[i] > T ? 1 : 0
  m = morph(morph(m, w, h, true), w, h, false)
  const border: number[] = []
  for (let x = 0; x < w; x++) border.push(x, (h - 1) * w + x)
  for (let y = 0; y < h; y++) border.push(y * w, y * w + w - 1)
  const outside = new Uint8Array(w * h)
  flood(m, w, h, border, (i) => !m[i]).forEach((i) => (outside[i] = 1))
  for (let i = 0; i < w * h; i++) if (!outside[i]) m[i] = 1
  const lab = new Uint8Array(w * h)
  let best: number[] = []
  for (let i = 0; i < w * h; i++) {
    if (m[i] && !lab[i]) {
      const comp = flood(m, w, h, [i], (j) => !!m[j] && !lab[j])
      comp.forEach((j) => (lab[j] = 1))
      if (comp.length > best.length) best = comp
    }
  }
  const mask = new Uint8Array(w * h)
  best.forEach((i) => (mask[i] = 1))

  // contour pixels
  const edge = new Uint8Array(w * h)
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      if (mask[i] && (!mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w])) edge[i] = 1
    }
  const thick = morph(morph(edge, w, h, true), w, h, true)

  // signed distance (chamfer 3-4, 2 pass)
  const INF = 1e9
  const dist = new Float64Array(w * h)
  for (let i = 0; i < w * h; i++) dist[i] = edge[i] ? 0 : INF
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      if (x > 0) dist[i] = Math.min(dist[i], dist[i - 1] + 3)
      if (y > 0) {
        dist[i] = Math.min(dist[i], dist[i - w] + 3)
        if (x > 0) dist[i] = Math.min(dist[i], dist[i - w - 1] + 4)
        if (x < w - 1) dist[i] = Math.min(dist[i], dist[i - w + 1] + 4)
      }
    }
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x
      if (x < w - 1) dist[i] = Math.min(dist[i], dist[i + 1] + 3)
      if (y < h - 1) {
        dist[i] = Math.min(dist[i], dist[i + w] + 3)
        if (x < w - 1) dist[i] = Math.min(dist[i], dist[i + w + 1] + 4)
        if (x > 0) dist[i] = Math.min(dist[i], dist[i + w - 1] + 4)
      }
    }
  let dmax = 1
  for (let i = 0; i < w * h; i++) if (!mask[i]) dmax = Math.max(dmax, dist[i])

  let bx0 = w, bx1 = 0, by0 = h, by1 = 0
  best.forEach((i) => { const x = i % w, y = (i / w) | 0; bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y) })
  const PAD = 18
  const box: Box = [Math.max(0, bx0 - PAD), Math.max(0, by0 - PAD), 0, 0]
  box[2] = Math.min(w, bx1 + PAD + 1) - box[0]
  box[3] = Math.min(h, by1 + PAD + 1) - box[1]
  const tight: Box = [bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1]

  const xray = url
  const maskUrl = toUrl(w, h, (d) => {
    for (let i = 0; i < w * h; i++) { const v = mask[i] ? 238 : 10; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255 }
  }, box)
  const contourUrl = toUrl(w, h, (d) => {
    for (let i = 0; i < w * h; i++) { const v = thick[i] ? 240 : 10; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255 }
  }, box)
  const sdfUrl = toUrl(w, h, (d) => {
    for (let i = 0; i < w * h; i++) {
      // 안쪽(음수) = 밝게, 바깥(양수) = 거리 따라 어둡게
      const t = mask[i] ? 0.86 + Math.min(1, dist[i] / 90) * 0.14 : 0.62 * (1 - Math.min(1, dist[i] / (dmax * 0.35)))
      const [r, g, b] = cmap(t)
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = 255
    }
  }, box)

  // contour sample points (좌/우 외곽, 일정 행 간격) — 도식용
  const y0 = by0, y1 = by1
  const pts: [number, number][] = []
  const rows = 11
  for (let r = 0; r <= rows; r++) {
    const y = Math.round(y0 + 6 + ((y1 - y0 - 12) * r) / rows)
    let l = -1, rr = -1
    for (let x = 0; x < w; x++) if (edge[y * w + x]) { if (l < 0) l = x; rr = x }
    if (l >= 0) { pts.push([l, y]); if (rr - l > 6) pts.push([rr, y]) }
  }
  const sc = document.createElement('canvas')
  sc.width = w; sc.height = h
  const s2 = sc.getContext('2d')!
  const id = s2.createImageData(w, h)
  for (let i = 0; i < w * h; i++) {
    if (thick[i]) { id.data[i * 4] = 40; id.data[i * 4 + 1] = 48; id.data[i * 4 + 2] = 56; id.data[i * 4 + 3] = 255 }
  }
  s2.putImageData(id, 0, 0)
  s2.fillStyle = '#1e6f5c'
  pts.forEach(([x, y]) => { s2.beginPath(); s2.arc(x, y, 7, 0, Math.PI * 2); s2.fill() })
  const samplesUrl = crop(sc, box)

  const silhouette = toUrl(w, h, (d) => {
    for (let i = 0; i < w * h; i++) { d[i * 4] = 170; d[i * 4 + 1] = 164; d[i * 4 + 2] = 150; d[i * 4 + 3] = mask[i] ? 255 : 0 }
  }, tight)

  return { xray, mask: maskUrl, contour: contourUrl, sdf: sdfUrl, samples: samplesUrl, silhouette, T, med, sig, w, h }
}

export function useDrrSteps(url: string, rawMin: number, rawMax: number) {
  const [steps, setSteps] = useState<DrrSteps | null>(null)
  useEffect(() => {
    let alive = true
    setSteps(null)
    computeSteps(url, rawMin, rawMax).then((s) => alive && setSteps(s)).catch(() => alive && setSteps(null))
    return () => { alive = false }
  }, [url, rawMin, rawMax])
  return steps
}
