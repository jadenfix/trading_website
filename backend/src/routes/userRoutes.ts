// backend/src/routes/userRoutes.ts
import { Router } from 'express'
import { validateBody } from '../middleware/validate'
import {
  signupSchema,
  loginSchema,
  updateSchema,
} from '../schemas/userSchemas'
import {
  createUser,
  login,
  whoami,
  getUserById,
  updateUser,
} from '../controllers/userController'
import { authenticate } from '../middleware/auth'

const router = Router()

// 1️⃣ Public: Sign up
router.post(
  '/',
  validateBody(signupSchema),
  createUser
)

// 2️⃣ Public: Log in
router.post(
  '/login',
  validateBody(loginSchema),
  login
)

// 3️⃣ Protected: “Who am I?”
router.get(
  '/me',
  authenticate,
  whoami
)

// 4️⃣ Public: Lookup any user by ID
router.get(
  '/:id',
  getUserById
)

// 5️⃣ Protected: Update a user
router.put(
  '/:id',
  authenticate,
  validateBody(updateSchema),
  updateUser
)

export default router