import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '../lib/supabaseClient';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!supabase) {
            setLoading(false);
            return;
        }

        // Initial session check
        const checkUser = async () => {
            try {
                const { data: { session }, error } = await supabase.auth.getSession();
                if (session) {
                    setUser(session.user);
                }
            } catch (err) {
                console.warn("Auth session check skipped (Guest Mode active or service unavailable):", err);
            }
            setLoading(false);
        };

        checkUser();

        // Listen for auth changes
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setUser(session?.user ?? null);
            setLoading(false);
        });

        return () => subscription?.unsubscribe();
    }, []);

    const login = async (email, password) => {
        if (!supabase) {
            console.log("Guardian Guest Mode: Logging in as", email);
            setUser({ email, guest: true });
            return { user: { email } };
        }
        const { data, error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        return data;
    };

    const loginWithGoogle = async () => {
        if (!supabase) {
            setUser({ email: "guest@institution.edu", guest: true });
            return;
        }
        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: window.location.origin
            }
        });
        if (error) throw error;
        return data;
    };

    const register = async (email, password, metadata = {}) => {
        if (!supabase) throw new Error("Authentication service is not configured.");
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
                data: metadata
            }
        });
        if (error) throw error;
        return data;
    };

    const logout = async () => {
        if (!supabase) {
            setUser(null);
            return;
        }
        const { error } = await supabase.auth.signOut();
        if (error) console.error('Sign out error:', error.message);
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, loading, login, loginWithGoogle, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
