// frontend/src/pages/HomePage.tsx
import React from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function HomePage() {
  const { user, logout } = useAuth()

  return (
    <div style={{ padding: '1rem' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between' }}>
        <h1>Welcome, {user?.name}!</h1>
        <button onClick={logout}>Log out</button>
      </header>

      <nav style={{ marginTop: '2rem' }}>
        <Link to="/backtest">Run a Backtest →</Link>
      </nav>
    </div>
  )
}