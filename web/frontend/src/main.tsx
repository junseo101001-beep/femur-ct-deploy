import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ResearchProvider } from './data/store'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <ResearchProvider>
        <App />
      </ResearchProvider>
    </BrowserRouter>
  </StrictMode>,
)
