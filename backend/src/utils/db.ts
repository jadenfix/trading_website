import mongoose from 'mongoose'

const MONGO_URI = process.env.MONGO_URI!
if (!MONGO_URI) {
  throw new Error('🛑 Missing MONGO_URI in environment')
}

export async function connectToDatabase(): Promise<void> {
  try {
    await mongoose.connect(MONGO_URI)
    console.log('✅ Connected to MongoDB')
  } catch (err) {
    console.error('❌ MongoDB connection error:', err)
    throw err
  }
}