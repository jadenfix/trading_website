import React, { useState } from 'react'
import './ResearchPage.css'

export default function ResearchPage() {
  const [searchQuery, setSearchQuery] = useState('')

  const upcomingFeatures = [
    'AI-powered news analysis and sentiment detection',
    'Real-time market data integration',
    'Custom research queries with natural language',
    'Automated report generation',
    'Integration with your trading strategies'
  ]

  const mockInsights = [
    {
      title: 'Market Volatility Alert',
      type: 'alert',
      content: 'Increased volatility detected in tech sector. Consider adjusting position sizes.',
      time: '2 hours ago',
      priority: 'high'
    },
    {
      title: 'NVDA Earnings Preview',
      type: 'research',
      content: 'Upcoming earnings announcement expected to impact semiconductor sector.',
      time: '4 hours ago',
      priority: 'medium'
    },
    {
      title: 'Fed Rate Decision Impact',
      type: 'analysis',
      content: 'Analysis of potential market reactions to upcoming Federal Reserve announcement.',
      time: '1 day ago',
      priority: 'medium'
    }
  ]

  return (
    <div className="research-page">
      <div className="research-header">
        <h1>🔍 Market Research</h1>
        <p>AI-powered market analysis and research tools</p>
      </div>

      <div className="research-search">
        <div className="search-container">
          <input
            type="text"
            placeholder="Ask anything about the markets... (Coming Soon)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
            disabled
          />
          <button className="search-button" disabled>
            <span>🤖</span>
            Research
          </button>
        </div>
      </div>

      <div className="research-content">
        <div className="insights-section">
          <h2>Latest Market Insights</h2>
          <p className="section-subtitle">AI-generated analysis and alerts (Preview)</p>
          
          <div className="insights-grid">
            {mockInsights.map((insight, index) => (
              <div key={index} className={`insight-card ${insight.priority}`}>
                <div className="insight-header">
                  <div className="insight-type">{insight.type}</div>
                  <div className="insight-time">{insight.time}</div>
                </div>
                <h3>{insight.title}</h3>
                <p>{insight.content}</p>
                <div className="insight-actions">
                  <button className="insight-action">View Details</button>
                  <button className="insight-action secondary">Add to Watchlist</button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="coming-soon-section">
          <h2>🚀 Coming Soon</h2>
          <p>Advanced AI research features powered by optimized LLMs</p>
          
          <div className="features-list">
            {upcomingFeatures.map((feature, index) => (
              <div key={index} className="feature-item">
                <span className="feature-icon">✨</span>
                <span className="feature-text">{feature}</span>
              </div>
            ))}
          </div>

          <div className="development-status">
            <h3>Development Status</h3>
            <div className="status-item">
              <span className="status-label">LLM Engine Optimization</span>
              <div className="status-bar">
                <div className="status-progress" style={{width: '75%'}}></div>
              </div>
              <span className="status-percent">75%</span>
            </div>
            <div className="status-item">
              <span className="status-label">News Data Integration</span>
              <div className="status-bar">
                <div className="status-progress" style={{width: '45%'}}></div>
              </div>
              <span className="status-percent">45%</span>
            </div>
            <div className="status-item">
              <span className="status-label">UI/UX Design</span>
              <div className="status-bar">
                <div className="status-progress" style={{width: '90%'}}></div>
              </div>
              <span className="status-percent">90%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 