// src/middleware/errorHandler.ts
import { Request, Response, NextFunction } from 'express'

export function errorHandler(
  err: any,
  _req: Request,
  res: Response,
  _next: NextFunction
) {
  console.error(err)                     // for your logs
  const status = err.status || 500       // if you throw { status: 400, message: ... }
  const message = err.message || 'Internal Server Error'
  res.status(status).json({ error: message })
}