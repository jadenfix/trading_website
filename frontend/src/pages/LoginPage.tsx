// frontend/src/pages/LoginPage.tsx
import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '' })
  const [error, setError] = useState<string|null>(null)

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await login(form.email, form.password)
      navigate('/')
    } catch (err: any) {
      setError(err.response?.data?.error || err.message)
    }
  }

  return (
    <form onSubmit={onSubmit}>
      <h2>Log In</h2>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      <label>
        Email
        <input name="email" type="email" value={form.email} onChange={onChange} required />
      </label>
      <label>
        Password
        <input name="password" type="password" value={form.password} onChange={onChange} required />
      </label>
      <button type="submit">Log In</button>
      <p>
        Don’t have an account? <Link to="/register">Register</Link>
      </p>
    </form>
  )
}