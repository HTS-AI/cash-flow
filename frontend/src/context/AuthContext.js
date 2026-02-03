import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

// Hardcoded users
const VALID_USERS = [
  { userId: 'HtsAI-testuser', password: 'HTStest@2025' },
  { userId: 'HtsAI', password: 'HTS@2025' }
];

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in
    const storedUser = localStorage.getItem('cashflow_user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const login = (userId, password) => {
    const validUser = VALID_USERS.find(
      (u) => u.userId === userId && u.password === password
    );

    if (validUser) {
      const userData = { userId: validUser.userId };
      setUser(userData);
      localStorage.setItem('cashflow_user', JSON.stringify(userData));
      return { success: true };
    }

    return { success: false, error: 'Invalid user ID or password' };
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('cashflow_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
