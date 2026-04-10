import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabaseClient';
import { getProfile, logActivity } from '../lib/supabaseDB';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState({ 
        id: 'dev-user-id', 
        email: 'dev@cybronites.local', 
        guest: true,
        user_metadata: { full_name: 'Lead Developer' }
    });
    const [profile, setProfile] = useState({
        username: 'DevAdmin',
        full_name: 'Lead Developer'
    });
    const [loading, setLoading] = useState(false);
    const [activeSessionId, setActiveSessionId] = useState(null);
    const [authError, setAuthError] = useState(null);

    const fetchProfile = useCallback(async (userId) => {
        if (!userId || !supabase) return;
        try {
            const p = await getProfile(userId);
            setProfile(p);
        } catch (err) {
            console.warn('Profile fetch error:', err);
        }
    }, []);

    useEffect(() => {
        if (!supabase) {
            setLoading(false);
            return;
        }

        // Check for OAuth error in URL (e.g., from failed Google login)
        const params = new URLSearchParams(window.location.search);
        const hashParams = new URLSearchParams(window.location.hash.replace('#', '?'));
        const errorDesc = params.get('error_description') || hashParams.get('error_description');
        if (errorDesc) {
            setAuthError(decodeURIComponent(errorDesc));
            // Clean the URL
            window.history.replaceState(null, '', window.location.pathname);
        }

        // Check existing session
        const checkUser = async () => {
            try {
                const { data: { session } } = await supabase.auth.getSession();
                if (session?.user) {
                    setUser(session.user);
                    await fetchProfile(session.user.id);
                }
            } catch (err) {
                console.warn("Session check failed:", err);
            }
            setLoading(false);
        };

        checkUser();

        // Listen for auth state changes (handles OAuth redirect automatically)
        const { data: { subscription } } = supabase.auth.onAuthStateChange(async (event, session) => {
            const u = session?.user ?? null;
            setUser(u);
            if (u) {
                await fetchProfile(u.id);
                if (event === 'SIGNED_IN') {
                    logActivity(u.id, 'LOGIN', { provider: u.app_metadata?.provider || 'email' });
                }
            } else {
                setProfile(null);
            }
            setLoading(false);
        });

        return () => subscription?.unsubscribe();
    }, [fetchProfile]);

    const login = async (email, password) => {
        if (!supabase) {
            setUser({ email, guest: true });
            return { user: { email } };
        }
        setAuthError(null);
        const { data, error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        logActivity(data.user.id, 'LOGIN', { method: 'password' });
        return data;
    };

    const loginWithGoogle = async () => {
        if (!supabase) {
            setUser({ email: "guest@institution.edu", guest: true });
            return;
        }
        setAuthError(null);
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
        setAuthError(null);
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: { data: metadata }
        });
        if (error) throw error;
        if (data.user) {
            logActivity(data.user.id, 'REGISTER', { email });
        }
        return data;
    };

    const logout = async () => {
        if (user && !user.guest) logActivity(user.id, 'LOGOUT');
        if (!supabase) {
            setUser(null);
            setProfile(null);
            return;
        }
        await supabase.auth.signOut();
        setUser(null);
        setProfile(null);
        setActiveSessionId(null);
    };

    const displayName = profile?.username
        || profile?.full_name
        || user?.user_metadata?.full_name
        || user?.user_metadata?.username
        || user?.email?.split('@')[0]
        || 'Researcher';

    return (
        <AuthContext.Provider value={{
            user, profile, loading, authError,
            login, loginWithGoogle, register, logout,
            displayName, activeSessionId, setActiveSessionId,
            refreshProfile: () => user && fetchProfile(user.id)
        }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
