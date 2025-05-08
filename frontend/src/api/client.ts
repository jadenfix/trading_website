// frontend/src/api/client.ts
import axios from 'axios'

const client = axios.create({
  baseURL: '/api',                     // <-- USE RELATIVE API PATH
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
})

client.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token')
  if (token && cfg.headers) {
    cfg.headers.Authorization = `Bearer ${token}`
  }
  return cfg
})

export default client