// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth }          from './context/AuthContext'
import HomePage                            from './pages/HomePage'
import BacktestPage                        from './pages/BacktestPage'
import LoginPage                           from './pages/LoginPage'
import RegisterPage                        from './pages/RegisterPage'

function PrivateRoute({ children }: { children: JSX.Element }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}

// src/App.tsx
export default function App() {
  return (
    <div>
      <h1>🛠️ Test App Loaded</h1>
    </div>
  )
}