// frontend/src/components/Ping.tsx
import React, { useEffect, useState } from 'react'
import axios from 'axios'

export function Ping() {
  const [pong, setPong] = useState<string>('Loading…')

  useEffect(() => {
    axios.get('/api/ping')
      .then(res => setPong(JSON.stringify(res.data)))
      .catch(err => setPong('Error: ' + err.message))
  }, [])

  return (
    <div style={{ marginTop: 20 }}>
      <h2>Backend Ping:</h2>
      <pre>{pong}</pre>
    </div>
  )
}