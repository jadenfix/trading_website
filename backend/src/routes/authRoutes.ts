// backend/src/routes/authRoutes.ts
import { Router } from 'express'
import { validateBody } from '../middleware/validate'
import { signupSchema, loginSchema } from '../schemas/userSchemas'
import { createUser, login } from '../controllers/userController'

const router = Router()

// POST /api/auth/register
router.post(
  '/register',
  validateBody(signupSchema),
  createUser
)

// POST /api/auth/login
router.post(
  '/login',
  validateBody(loginSchema),
  login
)

export default router