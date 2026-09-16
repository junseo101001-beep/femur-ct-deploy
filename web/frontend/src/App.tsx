import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { PAGES } from './data/research'
import Home from './pages/Home'
import Reconstruction from './pages/Reconstruction'
import Results from './pages/Results'
import Robustness from './pages/Robustness'
import Algorithm from './pages/Algorithm'
import Fallback from './pages/Fallback'
import Limitations from './pages/Limitations'

function ScrollTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

export default function App() {
  return (
    <div className="app">
      <ScrollTop />

      <nav className="nav">
        <div className="wrap nav-in">
          <NavLink to="/" className="nav-home" aria-label="홈">
            <svg width="17" height="17" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round">
              <path d="M2 8.2 8 2.8l6 5.4" />
              <path d="M3.8 7.4V13.2h8.4V7.4" />
            </svg>
          </NavLink>
          <div className="nav-links">
            {PAGES.map((p) => (
              <NavLink key={p.path} to={p.path} className={({ isActive }) => (isActive ? 'active' : '')}>
                {p.label}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/reconstruction" element={<Reconstruction />} />
        <Route path="/results" element={<Results />} />
        <Route path="/robustness" element={<Robustness />} />
        <Route path="/algorithm" element={<Algorithm />} />
        <Route path="/fallback" element={<Fallback />} />
        <Route path="/limitations" element={<Limitations />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </div>
  )
}
