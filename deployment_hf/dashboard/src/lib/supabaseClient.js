import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

const isConfigured = supabaseUrl && supabaseAnonKey && supabaseAnonKey !== 'YOUR_SUPABASE_ANON_KEY_HERE';

if (!isConfigured) {
  console.warn('Supabase credentials missing or using placeholders. Auth logic will be disabled.');
}

// Only create the client if we have a valid URL to prevent the "supabaseUrl is required" crash
export const supabase = isConfigured 
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;
