// src/models/user.ts

import mongoose, { Schema, Document } from 'mongoose'

/**
 * 1️⃣ Define a TS interface for the shape your controllers expect
 */
export interface IUser {
  id: string
  name: string
  email: string
  passwordHash: string
}

/**
 * 2️⃣ Define the Mongoose Document (adds MongoDB fields on top of our User)
 */
interface UserDoc extends Document {
  name: string
  email: string
  passwordHash: string
}

/**
 * 3️⃣ Build the Mongoose schema & model
 */
const userSchema = new Schema<UserDoc>(
  {
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true, lowercase: true },
    passwordHash: { type: String, required: true },
  },
  {
    toJSON: {
      virtuals: true,
      versionKey: false,
      transform(_, ret) {
        ret.id = ret._id.toString()
        delete ret._id
      },
    },
  }
)

const UserModel = mongoose.model<UserDoc>('User', userSchema)

/**
 * 4️⃣ Mock in-memory storage for development without MongoDB
 */
let mockUsers: IUser[] = []
let mockIdCounter = 1

const mockUserModel = {
  async create(data: Omit<IUser, 'id'>): Promise<IUser> {
    const user: IUser = {
      id: (mockIdCounter++).toString(),
      ...data
    }
    mockUsers.push(user)
    return user
  },

  async findByEmail(email: string): Promise<IUser | null> {
    return mockUsers.find(u => u.email === email) || null
  },

  async findById(id: string): Promise<IUser | null> {
    return mockUsers.find(u => u.id === id) || null
  },

  async findByIdAndUpdate(
    id: string,
    data: Partial<Omit<IUser, 'id'>>
  ): Promise<IUser | null> {
    const userIndex = mockUsers.findIndex(u => u.id === id)
    if (userIndex === -1) return null
    
    mockUsers[userIndex] = { ...mockUsers[userIndex], ...data }
    return mockUsers[userIndex]
  },
}

/**
 * 5️⃣ Re-export an object named "User" with the same methods you had before
 */
export const User = {
  async create(data: Omit<IUser, 'id'>): Promise<IUser> {
    // Use mock if MongoDB is not connected
    if (mongoose.connection.readyState !== 1) {
      console.log('📝 Using mock user storage (MongoDB not connected)')
      return mockUserModel.create(data)
    }
    
    const doc = new UserModel(data)
    await doc.save()
    return doc.toJSON() as IUser
  },

  async findByEmail(email: string): Promise<IUser | null> {
    if (mongoose.connection.readyState !== 1) {
      return mockUserModel.findByEmail(email)
    }
    
    const doc = await UserModel.findOne({ email })
    return doc ? (doc.toJSON() as IUser) : null
  },

  async findById(id: string): Promise<IUser | null> {
    if (mongoose.connection.readyState !== 1) {
      return mockUserModel.findById(id)
    }
    
    const doc = await UserModel.findById(id)
    return doc ? (doc.toJSON() as IUser) : null
  },

  async findByIdAndUpdate(
    id: string,
    data: Partial<Omit<IUser, 'id'>>
  ): Promise<IUser | null> {
    if (mongoose.connection.readyState !== 1) {
      return mockUserModel.findByIdAndUpdate(id, data)
    }
    
    const doc = await UserModel.findByIdAndUpdate(id, data, { new: true })
    return doc ? (doc.toJSON() as IUser) : null
  },
}
