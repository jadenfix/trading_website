// frontend/src/main.tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

console.log("🟢 React mounting…", document.getElementById('root'))
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {console.log('🟢 Inside <StrictMode>')}
    <App />
  </React.StrictMode>
)