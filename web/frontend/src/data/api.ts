/**
 * API base URL (배포층 설정 — 연구 코드와 무관).
 * - 로컬 dev: 미설정 → same-origin '/api/*' (vite proxy → localhost:8018)
 * - 배포: VITE_API_BASE_URL=https://<render-app>.onrender.com
 */
export const API_BASE: string = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
).replace(/\/$/, '')

export function api(path: string): string {
  return `${API_BASE}${path}`
}
