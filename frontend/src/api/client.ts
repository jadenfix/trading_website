// frontend/src/api/client.ts
import axios from 'axios';

const client = axios.create({
  // in dev a Vite proxy rewrites “/api” → http://localhost:8000
  baseURL: import.meta.env.VITE_API_URL || '/api',
  withCredentials: true            // cookies if you ever use them
});

export default client;   // <‑‑ default
export { client };       // <‑‑ named (so either import style works)