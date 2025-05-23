import express from 'express'
import cors from 'cors'
import userRoutes from './routes/userRoutes'
import authRoutes from './routes/authRoutes'
import { errorHandler } from './middleware/errorHandler'

export const app = express()
app.use(cors())
app.use(express.json())
app.use('/api/users', userRoutes)
app.use('/api/auth', authRoutes)
app.use(errorHandler)