/**
 * AI GUARDIAN — Supabase Database Service Layer
 * All database operations tied to authenticated user's ID.
 * Uses RLS (Row Level Security) so queries automatically scope to the current user.
 */
import { supabase } from './supabaseClient';

// ─── Cache Layer ───
const cache = new Map();
const CACHE_TTL = 30_000; // 30 seconds

function getCached(key) {
  const entry = cache.get(key);
  if (entry && Date.now() - entry.ts < CACHE_TTL) return entry.data;
  cache.delete(key);
  return null;
}

function setCache(key, data) {
  cache.set(key, { data, ts: Date.now() });
}

export function invalidateCache(prefix) {
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
}

// ─── Guard ───
function guard() {
  if (!supabase) throw new Error('Supabase not configured');
}

// ═══════════════════════════════════════════════════
// PROFILES
// ═══════════════════════════════════════════════════

export async function getProfile(userId) {
  guard();
  const cacheKey = `profile:${userId}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const { data, error } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single();

  if (error && error.code !== 'PGRST116') {
    console.error('Profile fetch error:', error);
    return null;
  }
  if (data) setCache(cacheKey, data);
  return data;
}

export async function updateProfile(userId, updates) {
  guard();
  const { data, error } = await supabase
    .from('profiles')
    .upsert({ id: userId, ...updates, updated_at: new Date().toISOString() })
    .select()
    .single();

  if (error) throw error;
  invalidateCache('profile:');
  return data;
}

// ═══════════════════════════════════════════════════
// TRAINING SESSIONS
// ═══════════════════════════════════════════════════

export async function createTrainingSession(userId, hyperparams = {}) {
  guard();
  const { data, error } = await supabase
    .from('training_sessions')
    .insert({
      user_id: userId,
      status: 'RUNNING',
      hyperparams
    })
    .select()
    .single();

  if (error) throw error;
  invalidateCache('sessions:');
  return data;
}

export async function updateTrainingSession(sessionId, updates) {
  guard();
  const { data, error } = await supabase
    .from('training_sessions')
    .update(updates)
    .eq('id', sessionId)
    .select()
    .single();

  if (error) throw error;
  invalidateCache('sessions:');
  return data;
}

export async function getTrainingSessions(userId, limit = 20) {
  guard();
  const cacheKey = `sessions:${userId}:${limit}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const { data, error } = await supabase
    .from('training_sessions')
    .select('*')
    .eq('user_id', userId)
    .order('started_at', { ascending: false })
    .limit(limit);

  if (error) throw error;
  setCache(cacheKey, data);
  return data || [];
}

// ═══════════════════════════════════════════════════
// EXPERIMENT LOGS (per-round results)
// ═══════════════════════════════════════════════════

export async function logExperimentRound(userId, sessionId, roundData) {
  guard();
  const { data, error } = await supabase
    .from('experiment_logs')
    .insert({
      user_id: userId,
      session_id: sessionId,
      round_number: roundData.round,
      accuracy: roundData.accuracy,
      loss: roundData.loss,
      clients_active: roundData.clientsActive,
      rejected_count: roundData.rejectedCount,
      blockchain_hash: roundData.blockchainHash,
      log_data: roundData.metadata || {}
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function getExperimentLogs(userId, sessionId = null, limit = 100) {
  guard();
  let query = supabase
    .from('experiment_logs')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (sessionId) query = query.eq('session_id', sessionId);

  const { data, error } = await query;
  if (error) throw error;
  return data || [];
}

// ═══════════════════════════════════════════════════
// ARCHITECTURE CONFIGS
// ═══════════════════════════════════════════════════

export async function saveArchitectureConfig(userId, name, config) {
  guard();
  const { data, error } = await supabase
    .from('architecture_configs')
    .insert({
      user_id: userId,
      name,
      config
    })
    .select()
    .single();

  if (error) throw error;
  invalidateCache('arch:');
  return data;
}

export async function getArchitectureConfigs(userId) {
  guard();
  const cacheKey = `arch:${userId}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const { data, error } = await supabase
    .from('architecture_configs')
    .select('*')
    .eq('user_id', userId)
    .order('updated_at', { ascending: false });

  if (error) throw error;
  setCache(cacheKey, data);
  return data || [];
}

export async function deleteArchitectureConfig(configId) {
  guard();
  const { error } = await supabase
    .from('architecture_configs')
    .delete()
    .eq('id', configId);

  if (error) throw error;
  invalidateCache('arch:');
}

// ═══════════════════════════════════════════════════
// LAB EXPERIMENTS
// ═══════════════════════════════════════════════════

export async function createLabExperiment(userId, code) {
  guard();
  const { data, error } = await supabase
    .from('lab_experiments')
    .insert({
      user_id: userId,
      code,
      status: 'RUNNING'
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function updateLabExperiment(experimentId, updates) {
  guard();
  const { data, error } = await supabase
    .from('lab_experiments')
    .update(updates)
    .eq('id', experimentId)
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function getLabExperiments(userId, limit = 20) {
  guard();
  const cacheKey = `lab:${userId}:${limit}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const { data, error } = await supabase
    .from('lab_experiments')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw error;
  setCache(cacheKey, data);
  return data || [];
}

// ═══════════════════════════════════════════════════
// ACTIVITY LOG
// ═══════════════════════════════════════════════════

export async function logActivity(userId, action, details = {}) {
  if (!supabase || !userId) return; // Fail silently for guest mode

  try {
    await supabase
      .from('activity_log')
      .insert({
        user_id: userId,
        action,
        details
      });
  } catch (err) {
    console.warn('Activity log failed (non-critical):', err.message);
  }
}

export async function getActivityLog(userId, limit = 50) {
  guard();
  const { data, error } = await supabase
    .from('activity_log')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) throw error;
  return data || [];
}

// ═══════════════════════════════════════════════════
// DASHBOARD STATS (aggregated)
// ═══════════════════════════════════════════════════

export async function getUserDashboardStats(userId) {
  guard();
  const cacheKey = `stats:${userId}`;
  const cached = getCached(cacheKey);
  if (cached) return cached;

  const [sessions, experiments, labExps] = await Promise.all([
    supabase.from('training_sessions').select('id, status, final_accuracy', { count: 'exact' }).eq('user_id', userId),
    supabase.from('experiment_logs').select('id', { count: 'exact' }).eq('user_id', userId),
    supabase.from('lab_experiments').select('id', { count: 'exact' }).eq('user_id', userId)
  ]);

  const stats = {
    totalSessions: sessions.count || 0,
    completedSessions: (sessions.data || []).filter(s => s.status === 'COMPLETE').length,
    bestAccuracy: Math.max(0, ...(sessions.data || []).map(s => s.final_accuracy || 0)),
    totalRounds: experiments.count || 0,
    totalLabExperiments: labExps.count || 0
  };

  setCache(cacheKey, stats);
  return stats;
}
