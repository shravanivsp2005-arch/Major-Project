create table if not exists public.civic_documents (
  collection text not null,
  id text not null,
  data jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key (collection, id)
);

create index if not exists civic_documents_collection_idx
  on public.civic_documents (collection);

alter table public.civic_documents enable row level security;

-- The FastAPI server uses the service-role key and performs access control itself.
-- Keep this table inaccessible to browser clients.
-- Passwords are managed by Supabase Auth; this table stores application profiles
-- and must never contain plaintext passwords.
