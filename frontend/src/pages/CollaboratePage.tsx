import React, { useState } from 'react'
import './CollaboratePage.css'

export default function CollaboratePage() {
  const [activeTab, setActiveTab] = useState('feed')

  const mockPosts = [
    {
      id: 1,
      author: 'AlgoTrader_Pro',
      avatar: '👤',
      time: '2 hours ago',
      content: 'Just backtested my new momentum strategy on crypto pairs. Getting 15% annual returns with decent Sharpe ratio. Anyone interested in collaborating?',
      likes: 12,
      comments: 5,
      tags: ['momentum', 'crypto', 'backtest']
    },
    {
      id: 2,
      author: 'QuantQueen',
      avatar: '👩‍💼',
      time: '4 hours ago',
      content: 'Found an interesting pattern in the VIX futures curve. Seems like we can predict short-term volatility spikes with 70% accuracy.',
      likes: 8,
      comments: 3,
      tags: ['vix', 'volatility', 'prediction']
    },
    {
      id: 3,
      author: 'DeepLearningTrader',
      avatar: '🤖',
      time: '1 day ago',
      content: 'Training a transformer model on earnings call transcripts. Early results show we can predict post-earnings drift with impressive accuracy.',
      likes: 24,
      comments: 12,
      tags: ['AI', 'earnings', 'NLP']
    }
  ]

  const mockLeaderboard = [
    { rank: 1, name: 'QuantMaster', returns: '+45.2%', sharpe: '2.14', avatar: '👑' },
    { rank: 2, name: 'AlgoWizard', returns: '+38.7%', sharpe: '1.89', avatar: '🧙‍♂️' },
    { rank: 3, name: 'TradingBot2000', returns: '+34.1%', sharpe: '1.76', avatar: '🤖' },
    { rank: 4, name: 'VolatilityKing', returns: '+29.8%', sharpe: '1.62', avatar: '⚡' },
    { rank: 5, name: 'CryptoNinja', returns: '+27.3%', sharpe: '1.54', avatar: '🥷' },
  ]

  return (
    <div className="collaborate-page">
      <div className="collaborate-header">
        <h1>👥 Collaborate</h1>
        <p>Connect with traders, share strategies, and learn from the community</p>
      </div>

      <div className="collaborate-tabs">
        <button 
          className={`tab-button ${activeTab === 'feed' ? 'active' : ''}`}
          onClick={() => setActiveTab('feed')}
        >
          Community Feed
        </button>
        <button 
          className={`tab-button ${activeTab === 'leaderboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('leaderboard')}
        >
          Leaderboard
        </button>
        <button 
          className={`tab-button ${activeTab === 'groups' ? 'active' : ''}`}
          onClick={() => setActiveTab('groups')}
        >
          Trading Groups
        </button>
      </div>

      <div className="collaborate-content">
        {activeTab === 'feed' && (
          <div className="feed-section">
            <div className="post-composer">
              <div className="composer-header">
                <span className="composer-avatar">👤</span>
                <input 
                  type="text" 
                  placeholder="Share your trading insights, strategies, or ask questions..."
                  className="composer-input"
                />
              </div>
              <div className="composer-actions">
                <button className="composer-button">📊 Add Chart</button>
                <button className="composer-button">📈 Share Strategy</button>
                <button className="post-button">Post</button>
              </div>
            </div>

            <div className="posts-list">
              {mockPosts.map(post => (
                <div key={post.id} className="post-card">
                  <div className="post-header">
                    <div className="post-author">
                      <span className="author-avatar">{post.avatar}</span>
                      <div className="author-info">
                        <div className="author-name">{post.author}</div>
                        <div className="post-time">{post.time}</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="post-content">
                    {post.content}
                  </div>
                  
                  <div className="post-tags">
                    {post.tags.map(tag => (
                      <span key={tag} className="post-tag">#{tag}</span>
                    ))}
                  </div>
                  
                  <div className="post-actions">
                    <button className="action-button">
                      <span>👍</span>
                      <span>{post.likes}</span>
                    </button>
                    <button className="action-button">
                      <span>💬</span>
                      <span>{post.comments}</span>
                    </button>
                    <button className="action-button">
                      <span>🔄</span>
                      <span>Share</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <div className="leaderboard-section">
            <div className="leaderboard-header">
              <h2>Top Performers This Month</h2>
              <p>Based on risk-adjusted returns and community engagement</p>
            </div>
            
            <div className="leaderboard-list">
              {mockLeaderboard.map(trader => (
                <div key={trader.rank} className="leaderboard-item">
                  <div className="rank-badge">#{trader.rank}</div>
                  <div className="trader-info">
                    <span className="trader-avatar">{trader.avatar}</span>
                    <div className="trader-details">
                      <div className="trader-name">{trader.name}</div>
                      <div className="trader-stats">
                        <span className="stat-item">Returns: {trader.returns}</span>
                        <span className="stat-item">Sharpe: {trader.sharpe}</span>
                      </div>
                    </div>
                  </div>
                  <button className="follow-button">Follow</button>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'groups' && (
          <div className="groups-section">
            <div className="groups-header">
              <h2>Trading Groups</h2>
              <p>Join specialized communities focused on different trading strategies</p>
            </div>
            
            <div className="groups-grid">
              <div className="group-card">
                <div className="group-icon">📊</div>
                <h3>Quantitative Trading</h3>
                <p>Advanced mathematical models and algorithmic strategies</p>
                <div className="group-stats">
                  <span>1,247 members</span>
                  <span>•</span>
                  <span>42 active today</span>
                </div>
                <button className="join-button">Join Group</button>
              </div>
              
              <div className="group-card">
                <div className="group-icon">₿</div>
                <h3>Crypto Strategies</h3>
                <p>Cryptocurrency trading, DeFi, and blockchain analysis</p>
                <div className="group-stats">
                  <span>892 members</span>
                  <span>•</span>
                  <span>28 active today</span>
                </div>
                <button className="join-button">Join Group</button>
              </div>
              
              <div className="group-card">
                <div className="group-icon">🤖</div>
                <h3>AI & Machine Learning</h3>
                <p>Neural networks, deep learning, and AI-driven trading</p>
                <div className="group-stats">
                  <span>1,534 members</span>
                  <span>•</span>
                  <span>67 active today</span>
                </div>
                <button className="join-button">Join Group</button>
              </div>
              
              <div className="group-card">
                <div className="group-icon">⚡</div>
                <h3>High Frequency Trading</h3>
                <p>Ultra-low latency strategies and market microstructure</p>
                <div className="group-stats">
                  <span>456 members</span>
                  <span>•</span>
                  <span>15 active today</span>
                </div>
                <button className="join-button">Join Group</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 