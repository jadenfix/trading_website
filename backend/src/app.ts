import express from 'express'
import cors from 'cors'
import userRoutes from './routes/userRoutes'
import authRoutes from './routes/authRoutes'
import backtestRoutes from './routes/backtestRoutes'
import { errorHandler } from './middleware/errorHandler'

export const app = express()
app.use(cors())
app.use(express.json())
app.get('/api/ping', (_req, res) => {
  res.json({ pong: true, mode: 'provider-free' })
})
app.use('/api/users', userRoutes)
app.use('/api/auth', authRoutes)
app.use('/api/backtest', backtestRoutes)
app.use(errorHandler)
