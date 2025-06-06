import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import './Navigation.css'

export default function Navigation() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/research', label: 'Research', icon: '🔍' },
    { path: '/backtest', label: 'Backtesting', icon: '📈' },
    { path: '/collaborate', label: 'Collaborate', icon: '👥' },
    { path: '/codegen', label: 'Code Gen', icon: '⚡' },
  ]

  return (
    <nav className="navigation">
      <div className="nav-brand">
        <h2>🚀 TradingPro</h2>
      </div>
      
      <div className="nav-links">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </Link>
        ))}
      </div>

      <div className="nav-user">
        <div className="user-info">
          <span className="user-avatar">👤</span>
          <span className="user-name">Trader</span>
        </div>
      </div>
    </nav>
  )
} 