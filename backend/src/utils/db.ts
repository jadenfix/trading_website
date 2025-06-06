import mongoose from 'mongoose';

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/trading_website';

// Cache the connection to prevent multiple connections in development
let isConnected = false;

export async function connectToDatabase(): Promise<void> {
  // Return early if already connected
  if (isConnected) {
    console.log('✅ Using existing MongoDB connection');
    return;
  }

  // Set Mongoose options
  const options = {
    serverSelectionTimeoutMS: 5000, // 5 seconds timeout
    socketTimeoutMS: 45000, // 45 seconds socket timeout
    family: 4, // Use IPv4, skip trying IPv6
  };

  try {
    await mongoose.connect(MONGO_URI, options);
    isConnected = true;
    console.log('✅ Successfully connected to MongoDB');
    
    // Log connection events
    mongoose.connection.on('connected', () => {
      console.log('✅ MongoDB connected');
    });
    
    mongoose.connection.on('error', (err) => {
      console.error('❌ MongoDB connection error:', err);
    });
    
    mongoose.connection.on('disconnected', () => {
      console.log('ℹ️ MongoDB disconnected');
      isConnected = false;
    });
    
    // Close the Mongoose connection when the Node process ends
    process.on('SIGINT', async () => {
      await mongoose.connection.close();
      console.log('ℹ️ MongoDB connection closed through app termination');
      process.exit(0);
    });
  } catch (err) {
    console.error('❌ MongoDB connection error:', err);
    console.log('⚠️  Running without database - using mock data for development');
    // Don't exit, just continue without database
    isConnected = false;
  }
}