import React, { useState } from 'react'
import './CodeGenPage.css'

export default function CodeGenPage() {
  const [activeTemplate, setActiveTemplate] = useState('strategy')
  const [prompt, setPrompt] = useState('')

  const templates = [
    {
      id: 'strategy',
      name: 'Trading Strategy',
      icon: '📈',
      description: 'Generate complete trading strategy implementations',
      examples: [
        'Create a mean reversion strategy for SPY',
        'Build a momentum strategy with RSI and MACD',
        'Generate a pairs trading algorithm for tech stocks'
      ]
    },
    {
      id: 'indicator',
      name: 'Technical Indicators',
      icon: '📊',
      description: 'Custom technical indicator implementations',
      examples: [
        'Create a custom volatility indicator',
        'Build a volume-weighted momentum oscillator',
        'Generate a multi-timeframe strength index'
      ]
    },
    {
      id: 'backtest',
      name: 'Backtest Framework',
      icon: '🔄',
      description: 'Backtesting and analysis boilerplate code',
      examples: [
        'Set up a backtest framework with Zipline',
        'Create performance analysis functions',
        'Build risk management modules'
      ]
    },
    {
      id: 'data',
      name: 'Data Pipeline',
      icon: '🗃️',
      description: 'Data acquisition and processing utilities',
      examples: [
        'Build a real-time data feed connector',
        'Create data cleaning and preprocessing tools',
        'Generate feature engineering pipelines'
      ]
    }
  ]

  const recentGenerations = [
    {
      id: 1,
      title: 'Bollinger Bands Mean Reversion Strategy',
      type: 'Trading Strategy',
      time: '2 hours ago',
      language: 'Python',
      lines: 156
    },
    {
      id: 2,
      title: 'Volume-Weighted Moving Average Indicator',
      type: 'Technical Indicator',
      time: '1 day ago',
      language: 'Python',
      lines: 89
    },
    {
      id: 3,
      title: 'Portfolio Risk Analysis Framework',
      type: 'Backtest Framework',
      time: '2 days ago',
      language: 'Python',
      lines: 234
    }
  ]

  const upcomingFeatures = [
    'Multi-language support (Python, C++, JavaScript)',
    'Integration with popular trading libraries (Zipline, Backtrader, ccxt)',
    'Custom strategy optimization and parameter tuning',
    'Automated documentation generation',
    'Code testing and validation frameworks',
    'Integration with your backtesting engine'
  ]

  return (
    <div className="codegen-page">
      <div className="codegen-header">
        <h1>⚡ Code Generation</h1>
        <p>AI-powered code generation for trading algorithms and analysis tools</p>
      </div>

      <div className="codegen-content">
        <div className="generator-section">
          <div className="template-selector">
            <h2>Select Template Type</h2>
            <div className="templates-grid">
              {templates.map(template => (
                <div 
                  key={template.id}
                  className={`template-card ${activeTemplate === template.id ? 'active' : ''}`}
                  onClick={() => setActiveTemplate(template.id)}
                >
                  <div className="template-icon">{template.icon}</div>
                  <h3>{template.name}</h3>
                  <p>{template.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="prompt-section">
            <div className="prompt-header">
              <h2>Describe Your Requirements</h2>
              <span className="coming-soon-badge">Coming Soon</span>
            </div>
            
            <div className="prompt-input-container">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder={`Example prompts for ${templates.find(t => t.id === activeTemplate)?.name}:\n\n${templates.find(t => t.id === activeTemplate)?.examples.join('\n')}\n\n(AI code generation coming soon...)`}
                className="prompt-input"
                rows={8}
                disabled
              />
              
              <div className="prompt-actions">
                <div className="generation-options">
                  <label className="option-item">
                    <input type="checkbox" disabled />
                    <span>Include documentation</span>
                  </label>
                  <label className="option-item">
                    <input type="checkbox" disabled />
                    <span>Add unit tests</span>
                  </label>
                  <label className="option-item">
                    <input type="checkbox" disabled />
                    <span>Optimize for performance</span>
                  </label>
                </div>
                
                <button className="generate-button" disabled>
                  <span>🤖</span>
                  Generate Code
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="sidebar">
          <div className="recent-section">
            <h3>Recent Generations</h3>
            <div className="recent-list">
              {recentGenerations.map(item => (
                <div key={item.id} className="recent-item">
                  <div className="recent-header">
                    <div className="recent-title">{item.title}</div>
                    <div className="recent-time">{item.time}</div>
                  </div>
                  <div className="recent-details">
                    <span className="recent-type">{item.type}</span>
                    <span className="recent-meta">{item.language} • {item.lines} lines</span>
                  </div>
                  <div className="recent-actions">
                    <button className="recent-action">View</button>
                    <button className="recent-action">Copy</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="features-section">
            <h3>🚀 Coming Soon</h3>
            <p>Advanced RAG-powered code generation features</p>
            
            <div className="features-list">
              {upcomingFeatures.map((feature, index) => (
                <div key={index} className="feature-item">
                  <span className="feature-icon">✨</span>
                  <span className="feature-text">{feature}</span>
                </div>
              ))}
            </div>

            <div className="development-status">
              <h4>Development Progress</h4>
              <div className="status-item">
                <span className="status-label">RAG Engine</span>
                <div className="status-bar">
                  <div className="status-progress" style={{width: '60%'}}></div>
                </div>
                <span className="status-percent">60%</span>
              </div>
              <div className="status-item">
                <span className="status-label">Code Templates</span>
                <div className="status-bar">
                  <div className="status-progress" style={{width: '80%'}}></div>
                </div>
                <span className="status-percent">80%</span>
              </div>
              <div className="status-item">
                <span className="status-label">UI Integration</span>
                <div className="status-bar">
                  <div className="status-progress" style={{width: '95%'}}></div>
                </div>
                <span className="status-percent">95%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 