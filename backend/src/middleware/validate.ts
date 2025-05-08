// backend/src/middleware/validate.ts
import { Request, Response, NextFunction } from 'express'
import { Schema } from 'joi'

export function validateBody(schema: Schema) {
  return (req: Request, res: Response, next: NextFunction): void => {
    // collect all errors, strip unknown props
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true,
    })

    if (error) {
      // join all Joi messages into one string
      const messages = error.details.map((d) => d.message)
      res.status(400).json({ error: messages.join(', ') })
      return
    }

    // replace req.body with the validated & sanitized value
    req.body = value
    next()
  }
}