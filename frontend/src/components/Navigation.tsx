import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import './Navigation.css'

export default function Navigation() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: 'chart-line' },
    { path: '/research', label: 'Research', icon: 'search' },
    { path: '/backtest', label: 'Backtesting', icon: 'analytics' },
    { path: '/collaborate', label: 'Collaborate', icon: 'users' },
    { path: '/codegen', label: 'Code Gen', icon: 'code' },
  ]

  return (
    <nav className="navigation">
      <div className="nav-brand">
        <div className="brand-logo">
          <div className="logo-icon">
            <div className="cube">
              <div className="cube-face front"></div>
              <div className="cube-face back"></div>
              <div className="cube-face right"></div>
              <div className="cube-face left"></div>
              <div className="cube-face top"></div>
              <div className="cube-face bottom"></div>
            </div>
          </div>
          <div className="brand-text">
            <h2>CHADWICK</h2>
            <span className="brand-subtitle">ASSET MANAGEMENT</span>
          </div>
        </div>
      </div>
      
      <div className="nav-links">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
          >
            <span className={`nav-icon icon-${item.icon}`}></span>
            <span className="nav-label">{item.label}</span>
          </Link>
        ))}
      </div>

      <div className="nav-user">
        <div className="user-info">
          <div className="user-avatar">
            <div className="avatar-circle"></div>
          </div>
          <div className="user-details">
            <span className="user-name">Portfolio Manager</span>
            <span className="user-status">Active</span>
          </div>
        </div>
      </div>
    </nav>
  )
} 