import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Connect4App from './Connect4.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Connect4App />
  </StrictMode>,
)
