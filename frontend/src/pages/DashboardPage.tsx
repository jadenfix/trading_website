import React from 'react'
import { Link } from 'react-router-dom'
import './DashboardPage.css'

export default function DashboardPage() {
  const quickStats = [
    { label: 'Total Strategies', value: '12', change: '+2', icon: '📊' },
    { label: 'Active Backtests', value: '3', change: '+1', icon: '⚡' },
    { label: 'Portfolio Value', value: '$125,430', change: '+5.2%', icon: '💰' },
    { label: 'Win Rate', value: '68%', change: '+3%', icon: '🎯' },
  ]

  const quickActions = [
    { 
      title: 'Run New Backtest', 
      description: 'Test your trading strategies against historical data',
      icon: '📈',
      link: '/backtest',
      color: 'blue'
    },
    { 
      title: 'Research Market', 
      description: 'AI-powered market analysis and news insights',
      icon: '🔍',
      link: '/research',
      color: 'green'
    },
    { 
      title: 'Join Community', 
      description: 'Collaborate with other traders and share strategies',
      icon: '👥',
      link: '/collaborate',
      color: 'purple'
    },
    { 
      title: 'Generate Code', 
      description: 'Auto-generate trading algorithms and boilerplate',
      icon: '⚡',
      link: '/codegen',
      color: 'orange'
    },
  ]

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Trading Dashboard</h1>
        <p>Welcome back! Here's what's happening with your trading today.</p>
      </div>

      <div className="stats-grid">
        {quickStats.map((stat, index) => (
          <div key={index} className="stat-card">
            <div className="stat-icon">{stat.icon}</div>
            <div className="stat-content">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
              <div className="stat-change positive">{stat.change}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="actions-section">
        <h2>Quick Actions</h2>
        <div className="actions-grid">
          {quickActions.map((action, index) => (
            <Link key={index} to={action.link} className={`action-card ${action.color}`}>
              <div className="action-icon">{action.icon}</div>
              <div className="action-content">
                <h3>{action.title}</h3>
                <p>{action.description}</p>
              </div>
              <div className="action-arrow">→</div>
            </Link>
          ))}
        </div>
      </div>

      <div className="recent-activity">
        <h2>Recent Activity</h2>
        <div className="activity-list">
          <div className="activity-item">
            <div className="activity-icon">📊</div>
            <div className="activity-content">
              <div className="activity-title">MAC_5_20 Backtest Completed</div>
              <div className="activity-time">2 hours ago</div>
            </div>
            <div className="activity-result positive">+12.4%</div>
          </div>
          <div className="activity-item">
            <div className="activity-icon">🔍</div>
            <div className="activity-content">
              <div className="activity-title">Market Research: NVDA Analysis</div>
              <div className="activity-time">4 hours ago</div>
            </div>
            <div className="activity-result">Completed</div>
          </div>
          <div className="activity-item">
            <div className="activity-icon">⚡</div>
            <div className="activity-content">
              <div className="activity-title">Generated Momentum Strategy</div>
              <div className="activity-time">Yesterday</div>
            </div>
            <div className="activity-result">Ready</div>
          </div>
        </div>
      </div>
    </div>
  )
} 