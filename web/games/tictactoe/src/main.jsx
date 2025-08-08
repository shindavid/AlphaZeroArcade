import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '../../shared/shared.css'
import TicTacToeApp from './TicTacToe.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <TicTacToeApp />
  </StrictMode>,
)
