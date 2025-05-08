// frontend/src/services/authService.ts
import client from '../api/client'

export interface SignupPayload { name: string; email: string; password: string }
export interface LoginPayload  { email: string; password: string }

/**
 * POST /api/auth/register
 */
export function signup(data: SignupPayload) {
  return client.post('/auth/register', data)
}

/**
 * POST /api/auth/login
 */
export function login(data: LoginPayload) {
  return client.post('/auth/login', data)
}