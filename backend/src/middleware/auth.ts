// src/middleware/auth.ts
import { Request, Response, NextFunction } from 'express'
import jwt from 'jsonwebtoken'

// extend Express’s Request so TS knows about `req.user`
export interface AuthRequest extends Request {
  user?: { id: string; email: string }
}

export function authenticate(
  req: AuthRequest,
  res: Response,
  next: NextFunction
): void {
  const auth = req.headers.authorization
  if (!auth?.startsWith('Bearer ')) {
    res.status(401).json({ message: 'No token provided' })
    return
  }

  const token = auth.slice(7)
  try {
    // verify returns `any`; we’ll cast it to our payload shape
    const payload = jwt.verify(token, process.env.JWT_SECRET!) as {
      id: string
      email: string
    }
    req.user = payload
    next()
  } catch {
    res.status(401).json({ message: 'Invalid token' })
  }
}