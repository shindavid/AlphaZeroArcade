import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../../shared/shared.css'
import ChessApp from './Chess.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ChessApp />
  </StrictMode>,
)
