// frontend/src/context/AuthContext.tsx

import React, {
    createContext,
    useContext,
    useState,
    useEffect,
  } from 'react'
  import type { ReactNode } from 'react'
  import { client } from '../api/client'
  import * as authService from '../services/authService'
  import * as userService from '../services/userService'
  
  interface User {
    id: string
    name: string
    email: string
  }
  
  interface AuthContextValue {
    token: string | null
    user: User | null
    setUser: React.Dispatch<React.SetStateAction<User | null>>
    login: (email: string, password: string) => Promise<void>
    signup: (name: string, email: string, password: string) => Promise<void>
    logout: () => void
  }
  
  const AuthContext = createContext<AuthContextValue | undefined>(undefined)
  
  export function AuthProvider({ children }: { children: ReactNode }) {
    const [token, setToken] = useState<string | null>(
      () => localStorage.getItem('token')
    )
    const [user, setUser] = useState<User | null>(() => {
      const u = localStorage.getItem('user')
      return u ? JSON.parse(u) : null
    })
  
    useEffect(() => {
      if (token) {
        localStorage.setItem('token', token)
        client.defaults.headers.common.Authorization = `Bearer ${token}`
      } else {
        localStorage.removeItem('token')
        delete client.defaults.headers.common.Authorization
      }
    }, [token])
  
    useEffect(() => {
      if (user) localStorage.setItem('user', JSON.stringify(user))
      else localStorage.removeItem('user')
    }, [user])

    // Validate token on startup
    useEffect(() => {
      const validateToken = async () => {
        if (token && !user) {
          try {
            const me = (await userService.fetchMe()).data
            setUser(me)
          } catch (error: any) {
            // Token is invalid (401) or server error, clear authentication silently
            if (error.response?.status === 401) {
              console.log('🔄 Clearing expired authentication token')
            } else {
              console.log('⚠️ Authentication validation failed, clearing token')
            }
            setToken(null)
            setUser(null)
          }
        }
      }
      
      // Only validate if we have a token but no user data
      if (token && !user) {
        validateToken()
      }
    }, [token, user])
  
      const login = async (email: string, password: string) => {
    try {
      const res = await authService.login({ email, password })
      setToken(res.data.token)
      const me = (await userService.fetchMe()).data
      setUser(me)
    } catch (error) {
      // Clear any invalid tokens
      setToken(null)
      setUser(null)
      throw error
    }
  }

  const signup = async (name: string, email: string, password: string) => {
    try {
      const res = await authService.signup({ name, email, password })
      setToken(res.data.token)
      const me = (await userService.fetchMe()).data
      setUser(me)
    } catch (error) {
      // Clear any invalid tokens
      setToken(null)
      setUser(null)
      throw error
    }
  }
  
    const logout = () => {
      setToken(null)
      setUser(null)
    }
  
    return (
      <AuthContext.Provider value={{ token, user, setUser, login, signup, logout }}>
        {children}
      </AuthContext.Provider>
    )
  }
  
  export function useAuth(): AuthContextValue {
    const ctx = useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be inside AuthProvider')
    return ctx
  }