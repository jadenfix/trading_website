// frontend/src/context/AuthContext.tsx

import React, {
    createContext,
    useContext,
    useState,
    ReactNode,
    useEffect,
  } from 'react'
  import client from '../api/client'
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
  
    const login = async (email: string, password: string) => {
      const res = await authService.login({ email, password })
      setToken(res.data.token)
      const me = (await userService.fetchMe()).data
      setUser(me)
    }
  
    const signup = async (name: string, email: string, password: string) => {
      const res = await authService.signup({ name, email, password })
      setToken(res.data.token)
      const me = (await userService.fetchMe()).data
      setUser(me)
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