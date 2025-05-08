// frontend/src/services/userService.ts
import client from '../api/client'

/**
 * GET /api/users/me
 */
export function fetchMe() {
  return client.get<{ id: string; name: string; email: string }>('/users/me')
}

/**
 * PUT /api/users/:id
 */
export function updateProfile(
  id: string,
  payload: { name?: string; email?: string }
) {
  return client.put(`/users/${id}`, payload)
}