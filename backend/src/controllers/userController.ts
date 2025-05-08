// backend/src/controllers/userController.ts

import { Request, Response, NextFunction } from 'express'
import bcrypt from 'bcrypt'
import * as jwt from 'jsonwebtoken'
import { User, IUser } from '../models/user'     // ← import the new types

// pull these from .env
const JWT_SECRET: string = process.env.JWT_SECRET!
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1h'
if (!JWT_SECRET) {
  throw new Error('🛑 Missing JWT_SECRET in environment')
}

// extend Request to carry our JWT payload
export interface AuthRequest extends Request {
  user?: { id: string; email: string }
}

/**
 * 1️⃣  Sign up: create a new user + return a token
 */
export async function createUser(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { name, email, password } = req.body as {
      name?: string
      email?: string
      password?: string
    }

    if (!name || !email || !password) {
      res.status(400).json({ error: 'Missing name, email or password' })
      return
    }

    if (await User.findByEmail(email)) {
      res.status(409).json({ error: 'Email already in use' })
      return
    }

    const passwordHash = await bcrypt.hash(password, 10)

    // ← This now uses your Mongoose-backed create(
    const user = await User.create({
      name,
      email,
      passwordHash,
    } as Omit<IUser, 'id'>)

    const token = jwt.sign(
      { id: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN } as jwt.SignOptions
    )

    res.status(201).json({
      id: user.id,
      name: user.name,
      email: user.email,
      token,
    })
  } catch (err) {
    next(err)
  }
}

/**
 * 2️⃣  Log in: verify creds → return a token
 */
export async function login(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { email, password } = req.body as {
      email?: string
      password?: string
    }

    if (!email || !password) {
      res.status(400).json({ error: 'Missing email or password' })
      return
    }

    const user = await User.findByEmail(email)
    if (!user) {
      res.status(401).json({ error: 'Invalid credentials' })
      return
    }

    const match = await bcrypt.compare(password, user.passwordHash)
    if (!match) {
      res.status(401).json({ error: 'Invalid credentials' })
      return
    }

    const token = jwt.sign(
      { id: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN } as jwt.SignOptions
    )

    res.json({ token })
  } catch (err) {
    next(err)
  }
}

/**
 * 3️⃣  “Who am I?”: return the currently‐logged‐in user
 */
export async function whoami(
  req: AuthRequest,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    if (!req.user) {
      res.status(401).json({ error: 'Not authenticated' })
      return
    }
    const me = await User.findById(req.user.id)
    if (!me) {
      res.status(404).json({ error: 'Not found' })
      return
    }
    res.json({ id: me.id, name: me.name, email: me.email })
  } catch (err) {
    next(err)
  }
}

/**
 * 4️⃣  Lookup any user by ID (public)
 */
export async function getUserById(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const user = await User.findById(req.params.id)
    if (!user) {
      res.status(404).json({ error: 'Not found' })
      return
    }
    res.json({ id: user.id, name: user.name, email: user.email })
  } catch (err) {
    next(err)
  }
}

/**
 * 5️⃣  Update a user (protected)
 */
export async function updateUser(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const user = await User.findByIdAndUpdate(
      req.params.id,
      req.body as Partial<Omit<IUser, 'id'>>
    )
    if (!user) {
      res.status(404).json({ error: 'Not found' })
      return
    }
    res.json({ id: user.id, name: user.name, email: user.email })
  } catch (err) {
    next(err)
  }
}