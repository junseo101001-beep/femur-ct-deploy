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
          <NavLink to="/" className="nav-brand">FEMUR / XR</NavLink>
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
