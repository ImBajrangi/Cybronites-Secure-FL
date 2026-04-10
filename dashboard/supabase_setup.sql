-- ╔══════════════════════════════════════════════════════════════╗
-- ║  AI GUARDIAN — Supabase Database Schema                     ║
-- ║  Run this in Supabase SQL Editor (Dashboard > SQL Editor)   ║
-- ╚══════════════════════════════════════════════════════════════╝

-- ─── 1. PROFILES TABLE ───
-- Auto-created on signup via trigger
create table if not exists profiles (
  id uuid references auth.users on delete cascade not null primary key,
  updated_at timestamp with time zone default now(),
  username text unique,
  full_name text,
  avatar_url text,
  role text default 'researcher',
  institution text,
  constraint username_length check (char_length(username) >= 3)
);

alter table profiles enable row level security;

create policy "Public profiles are viewable by everyone." on profiles
  for select using (true);
create policy "Users can insert their own profile." on profiles
  for insert with check (auth.uid() = id);
create policy "Users can update their own profile." on profiles
  for update using (auth.uid() = id);

-- Trigger: auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, full_name, avatar_url, username)
  values (
    new.id,
    new.raw_user_meta_data->>'full_name',
    new.raw_user_meta_data->>'avatar_url',
    coalesce(new.raw_user_meta_data->>'username', split_part(new.email, '@', 1))
  );
  return new;
end;
$$ language plpgsql security definer;

-- Drop existing trigger if any, then recreate
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- ─── 2. TRAINING SESSIONS ───
-- Logs every federated training session initiated by a user
create table if not exists training_sessions (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users on delete cascade not null,
  started_at timestamp with time zone default now(),
  ended_at timestamp with time zone,
  status text default 'RUNNING',  -- RUNNING | COMPLETE | ERROR
  rounds_completed integer default 0,
  final_accuracy real,
  final_loss real,
  hyperparams jsonb default '{}',
  metadata jsonb default '{}'
);

alter table training_sessions enable row level security;

create policy "Users can view own sessions" on training_sessions
  for select using (auth.uid() = user_id);
create policy "Users can create own sessions" on training_sessions
  for insert with check (auth.uid() = user_id);
create policy "Users can update own sessions" on training_sessions
  for update using (auth.uid() = user_id);

-- ─── 3. EXPERIMENT LOGS ───
-- Tracks individual round/experiment results within a session
create table if not exists experiment_logs (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users on delete cascade not null,
  session_id uuid references training_sessions on delete cascade,
  created_at timestamp with time zone default now(),
  round_number integer,
  accuracy real,
  loss real,
  clients_active integer,
  rejected_count integer default 0,
  blockchain_hash text,
  log_data jsonb default '{}'
);

alter table experiment_logs enable row level security;

create policy "Users can view own experiments" on experiment_logs
  for select using (auth.uid() = user_id);
create policy "Users can create own experiments" on experiment_logs
  for insert with check (auth.uid() = user_id);

-- ─── 4. ARCHITECTURE CONFIGS ───
-- Saves model architecture configurations per user
create table if not exists architecture_configs (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users on delete cascade not null,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now(),
  name text not null default 'Untitled Model',
  config jsonb not null default '{}',
  is_active boolean default false
);

alter table architecture_configs enable row level security;

create policy "Users can view own configs" on architecture_configs
  for select using (auth.uid() = user_id);
create policy "Users can create own configs" on architecture_configs
  for insert with check (auth.uid() = user_id);
create policy "Users can update own configs" on architecture_configs
  for update using (auth.uid() = user_id);
create policy "Users can delete own configs" on architecture_configs
  for delete using (auth.uid() = user_id);

-- ─── 5. LAB EXPERIMENTS ───
-- Tracks code laboratory experiments and their results
create table if not exists lab_experiments (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users on delete cascade not null,
  created_at timestamp with time zone default now(),
  code text,
  output text,
  status text default 'RUNNING',  -- RUNNING | COMPLETE | ERROR
  execution_time_ms integer,
  metadata jsonb default '{}'
);

alter table lab_experiments enable row level security;

create policy "Users can view own lab experiments" on lab_experiments
  for select using (auth.uid() = user_id);
create policy "Users can create own lab experiments" on lab_experiments
  for insert with check (auth.uid() = user_id);
create policy "Users can update own lab experiments" on lab_experiments
  for update using (auth.uid() = user_id);

-- ─── 6. USER ACTIVITY LOG ───
-- General purpose activity tracking for analytics
create table if not exists activity_log (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users on delete cascade not null,
  created_at timestamp with time zone default now(),
  action text not null,           -- LOGIN | LOGOUT | START_TRAINING | VIEW_CHANGE | etc.
  details jsonb default '{}'
);

alter table activity_log enable row level security;

create policy "Users can view own activity" on activity_log
  for select using (auth.uid() = user_id);
create policy "Users can log own activity" on activity_log
  for insert with check (auth.uid() = user_id);

-- ─── 7. INDEXES FOR PERFORMANCE ───
create index if not exists idx_training_sessions_user on training_sessions(user_id);
create index if not exists idx_experiment_logs_user on experiment_logs(user_id);
create index if not exists idx_experiment_logs_session on experiment_logs(session_id);
create index if not exists idx_architecture_configs_user on architecture_configs(user_id);
create index if not exists idx_lab_experiments_user on lab_experiments(user_id);
create index if not exists idx_activity_log_user on activity_log(user_id);
create index if not exists idx_activity_log_created on activity_log(created_at desc);
