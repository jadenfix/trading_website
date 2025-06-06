import React, { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useNavigate, Link } from 'react-router-dom'

export default function RegisterPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string|null>(null)
  const { signup } = useAuth()
  const nav = useNavigate()

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      await signup(name, email, password)
      nav('/')
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Registration failed')
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <h2>Register</h2>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      <input value={name} onChange={e=>setName(e.target.value)} placeholder="Name" required />
      <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email" type="email" required />
      <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password" required />
      <button type="submit">Sign Up</button>
      <p>
        Already have an account? <Link to="/login">Login</Link>
      </p>
    </form>
  )
}