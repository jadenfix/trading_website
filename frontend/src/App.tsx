// frontend/src/App.tsx
import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import Navigation from './components/Navigation'
import DashboardPage from './pages/DashboardPage'
import ResearchPage from './pages/ResearchPage'
import BacktestPage from './pages/BacktestPage'
import CSVBacktestPage from './pages/CSVBacktestPage'
import CollaboratePage from './pages/CollaboratePage'
import CodeGenPage from './pages/CodeGenPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import './App.css'

// Simple component that bypasses authentication for development
const PrivateRoute: React.FC<{ children: React.ReactElement }> = ({ children }) => {
  // Bypass authentication for development - just render the children
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app">
          <Navigation />
          <main className="main-content">
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/" element={
                <PrivateRoute>
                  <DashboardPage />
                </PrivateRoute>
              } />
              <Route path="/research" element={
                <PrivateRoute>
                  <ResearchPage />
                </PrivateRoute>
              } />
              <Route path="/backtest" element={
                <PrivateRoute>
                  <BacktestPage />
                </PrivateRoute>
              } />
              <Route path="/csv-backtest" element={
                <PrivateRoute>
                  <CSVBacktestPage />
                </PrivateRoute>
              } />
              <Route path="/collaborate" element={
                <PrivateRoute>
                  <CollaboratePage />
                </PrivateRoute>
              } />
              <Route path="/codegen" element={
                <PrivateRoute>
                  <CodeGenPage />
                </PrivateRoute>
              } />
            </Routes>
          </main>
        </div>
      </Router>
    </AuthProvider>
  )
}