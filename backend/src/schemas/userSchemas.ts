import Joi from 'joi'

export const signupSchema = Joi.object({
  name: Joi.string().max(100).required().messages({
    'string.empty': `"name" is required`,
    'string.max':   `"name" must be at most 100 characters`,
  }),
  email: Joi.string().email().required().messages({
    'string.empty': `"email" is required`,
    'string.email': `"email" must be a valid email`,
  }),
  password: Joi.string().min(6).required().messages({
    'string.empty': `"password" is required`,
    'string.min':   `"password" must be at least 6 characters`,
  }),
})

export const loginSchema = Joi.object({
  email: Joi.string().email().required().messages({
    'string.empty': `"email" is required`,
    'string.email': `"email" must be a valid email`,
  }),
  password: Joi.string().required().messages({
    'string.empty': `"password" is required`,
  }),
})

export const updateSchema = Joi.object({
  name: Joi.string().max(100).messages({
    'string.empty': `"name" cannot be empty`,
    'string.max':   `"name" must be at most 100 characters`,
  }),
  email: Joi.string().email().messages({
    'string.email': `"email" must be a valid email`,
  }),
})
  .min(1)
  .messages({
    'object.min': `"value" must have at least one of [name, email]`,
  })