// frontend/src/api/client.ts
import axios from 'axios'

export const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  // …any other defaults…
})

export default client