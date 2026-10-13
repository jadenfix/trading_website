// backend/src/index.ts

import dotenv from 'dotenv'
dotenv.config() // ← Load .env as early as possible

import { app } from './app'
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
  if (process.env.SKIP_DB !== '1') {
    try {
      await connectToDatabase()
    } catch (err) {
      console.log('⚠️  Continuing without database connection')
    }
  } else {
    console.log('ℹ️  Database intentionally skipped for provider-free fixture mode')
  }

  // Start the shared app exported for real HTTP integration tests.
  const port = Number(process.env.PORT ?? 8000)
  app.listen(port, () => {
    console.log(`🚀 Backend listening on port ${port}`)
  })
}

main().catch(err => {
  console.error('❌ Failed to start server:', err)
  process.exit(1)
})
