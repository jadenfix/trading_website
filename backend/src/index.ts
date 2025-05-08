// backend/src/index.ts

import dotenv from 'dotenv'
dotenv.config() // ← Load .env as early as possible

import express, { Request, Response, NextFunction } from 'express'
import cors from 'cors'

import authRoutes from './routes/authRoutes'
import userRoutes from './routes/userRoutes'
import backtestRoutes from './routes/backtestRoutes'    // ← ← ←
import { errorHandler } from './middleware/errorHandler'
import { connectToDatabase } from './utils/db'

// Sanity-check your secrets
console.log(
  '🔑 JWT_SECRET:',
  process.env.JWT_SECRET
    ? process.env.JWT_SECRET.slice(0, 5) + '…'
    : '[missing]'
)

async function main() {
  // 1️⃣ Connect to the database (or stub)
  await connectToDatabase()

  // 2️⃣ Create the Express app
  const app = express()

  // 3️⃣ Core middleware
  app.use(cors())
  app.use(express.json())

  // 4️⃣ Health-check
  app.get('/api/ping', (_req: Request, res: Response) => {
    res.json({ pong: true })
  })

  // 5️⃣ Mount auth routes (public)
  app.use('/api/auth', authRoutes)

  // 7️⃣ Mount backtest routes
  app.use('/api/backtest', backtestRoutes)

  // 6️⃣ Mount user routes (some public, some protected)
  app.use('/api/users', userRoutes)


  // 8️⃣ Global error handler (must come last)
  app.use(errorHandler as any)

  // 9️⃣ Start listening
  const port = Number(process.env.PORT ?? 8000)
  app.listen(port, () => {
    console.log(`🚀 Backend listening on port ${port}`)
  })
}

main().catch(err => {
  console.error('❌ Failed to start server:', err)
  process.exit(1)
})