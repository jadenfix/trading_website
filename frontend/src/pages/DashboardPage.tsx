import React from 'react'
import { Link } from 'react-router-dom'
import './DashboardPage.css'

export default function DashboardPage() {
  const quickStats = [
    { label: 'Assets Under Management', value: '$2.4B', change: '+12.3%', trend: 'up', icon: 'portfolio' },
    { label: 'Active Strategies', value: '47', change: '+8', trend: 'up', icon: 'strategies' },
    { label: 'Alpha Generation', value: '23.7%', change: '+4.2%', trend: 'up', icon: 'alpha' },
    { label: 'Sharpe Ratio', value: '2.84', change: '+0.3', trend: 'up', icon: 'risk' },
  ]

  const quickActions = [
    { 
      title: 'Strategy Backtesting', 
      description: 'High-performance quantitative analysis and risk assessment',
      link: '/backtest',
      gradient: 'blue',
      icon: 'analytics'
    },
    { 
      title: 'Market Intelligence', 
      description: 'AI-driven market analysis and predictive modeling',
      link: '/research',
      gradient: 'green',
      icon: 'intelligence'
    },
    { 
      title: 'Institutional Network', 
      description: 'Connect with elite portfolio managers and analysts',
      link: '/collaborate',
      gradient: 'purple',
      icon: 'network'
    },
    { 
      title: 'Algorithm Development', 
      description: 'Advanced quantitative model generation and optimization',
      link: '/codegen',
      gradient: 'gold',
      icon: 'development'
    },
  ]

  const recentActivity = [
    {
      type: 'execution',
      title: 'Long/Short Equity Strategy Deployment',
      description: 'Successfully deployed $150M in high-conviction positions',
      time: '2 hours ago',
      result: '+2.4%',
      status: 'success'
    },
    {
      type: 'analysis',
      title: 'Volatility Arbitrage Model Update',
      description: 'Enhanced VIX surface analysis with ML optimization',
      time: '4 hours ago',
      result: 'Completed',
      status: 'completed'
    },
    {
      type: 'risk',
      title: 'Portfolio Risk Assessment',
      description: 'Comprehensive stress testing across all positions',
      time: '6 hours ago',
      result: 'VAR: -1.2%',
      status: 'completed'
    },
    {
      type: 'research',
      title: 'Sector Rotation Analysis',
      description: 'Identified emerging opportunities in technology sector',
      time: '1 day ago',
      result: 'High Conviction',
      status: 'actionable'
    }
  ]

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div className="header-content">
          <h1>Portfolio Command Center</h1>
          <p className="header-subtitle">Real-time institutional-grade analytics and execution platform</p>
        </div>
        <div className="header-metrics">
          <div className="live-indicator">
            <div className="pulse"></div>
            <span>LIVE MARKET DATA</span>
          </div>
        </div>
      </div>

      <div className="stats-grid">
        {quickStats.map((stat, index) => (
          <div key={index} className="stat-card">
            <div className="stat-header">
              <div className={`stat-icon icon-${stat.icon}`}></div>
              <div className={`stat-trend trend-${stat.trend}`}>
                <span className="trend-arrow"></span>
                <span className="trend-value">{stat.change}</span>
              </div>
            </div>
            <div className="stat-content">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="actions-section">
        <h2>Strategic Operations</h2>
        <div className="actions-grid">
          {quickActions.map((action, index) => (
            <Link key={index} to={action.link} className={`action-card gradient-${action.gradient}`}>
              <div className="action-background"></div>
              <div className={`action-icon icon-${action.icon}`}></div>
              <div className="action-content">
                <h3>{action.title}</h3>
                <p>{action.description}</p>
              </div>
              <div className="action-arrow">
                <div className="arrow-line"></div>
                <div className="arrow-head"></div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      <div className="activity-section">
        <h2>Market Operations Log</h2>
        <div className="activity-container">
          {recentActivity.map((activity, index) => (
            <div key={index} className="activity-item">
              <div className="activity-timeline">
                <div className={`timeline-dot dot-${activity.type}`}></div>
                <div className="timeline-line"></div>
              </div>
              <div className="activity-content">
                <div className="activity-header">
                  <h4>{activity.title}</h4>
                  <div className="activity-meta">
                    <span className="activity-time">{activity.time}</span>
                    <span className={`activity-status status-${activity.status}`}>{activity.result}</span>
                  </div>
                </div>
                <p className="activity-description">{activity.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
} 