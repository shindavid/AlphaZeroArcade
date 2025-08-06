import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Connect4App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Connect4App />
  </StrictMode>,
)
