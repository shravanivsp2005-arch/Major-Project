-- ==============================================================================
-- CivicPulse — Supabase PostgreSQL Schema & Relational Views
-- ==============================================================================
-- Run this script in the Supabase SQL Editor (https://supabase.com/dashboard/project/dgdtwrlnbnrmlkjsmdcu/sql)
-- This creates the core JSONB document store and auto-extracting Relational Views
-- so you can view Complaints, Users, AI Predictions, and Reports in the Table Editor.

-- 1. Core Document Store Table
CREATE TABLE IF NOT EXISTS public.civic_documents (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (collection, id)
);

-- Indices for rapid indexing & filtering
CREATE INDEX IF NOT EXISTS idx_civic_documents_collection ON public.civic_documents (collection);
CREATE INDEX IF NOT EXISTS idx_civic_documents_id ON public.civic_documents (id);
CREATE INDEX IF NOT EXISTS idx_civic_documents_data_gin ON public.civic_documents USING GIN (data);

-- 2. Complaints Relational View
-- Displays all citizen reports with individual relational columns in the Supabase Table Editor
CREATE OR REPLACE VIEW public.complaints_view AS
SELECT
    id,
    data->>'tracking_id' AS tracking_id,
    data->>'title' AS title,
    data->>'description' AS description,
    data->>'department' AS department,
    data->>'category' AS category,
    data->>'priority' AS priority,
    data->>'status' AS status,
    (data->>'ai_confidence')::NUMERIC AS ai_confidence,
    data->'ai_prediction'->>'reasoning' AS ai_reasoning,
    data->'ai_prediction'->>'source' AS ai_source,
    data->'location'->>'ward' AS ward,
    data->'location'->>'address' AS address,
    (data->'location'->>'lat')::NUMERIC AS latitude,
    (data->'location'->>'lng')::NUMERIC AS longitude,
    data->>'citizen_name' AS citizen_name,
    data->>'citizen_id' AS citizen_id,
    data->>'assigned_officer_name' AS assigned_officer,
    (data->>'is_duplicate')::BOOLEAN AS is_duplicate,
    data->>'duplicate_group_id' AS duplicate_group_id,
    (data->>'is_demo')::BOOLEAN AS is_demo,
    (data->>'created_at')::TIMESTAMPTZ AS created_at
FROM public.civic_documents
WHERE collection = 'complaints'
ORDER BY (data->>'created_at')::TIMESTAMPTZ DESC;

-- 3. Users Relational View
CREATE OR REPLACE VIEW public.users_view AS
SELECT
    id,
    data->>'name' AS name,
    data->>'email' AS email,
    data->>'role' AS role,
    data->>'department' AS department,
    data->>'phone' AS phone,
    data->>'language' AS language,
    (data->>'is_demo')::BOOLEAN AS is_demo,
    (data->>'created_at')::TIMESTAMPTZ AS created_at
FROM public.civic_documents
WHERE collection = 'users'
ORDER BY (data->>'created_at')::TIMESTAMPTZ DESC;

-- 4. AI Predictions Relational View
CREATE OR REPLACE VIEW public.ai_predictions_view AS
SELECT
    id,
    data->>'user_id' AS user_id,
    data->'prediction'->>'detected_issue' AS detected_issue,
    data->'prediction'->>'department' AS department,
    data->'prediction'->>'category' AS category,
    data->'prediction'->>'priority' AS priority,
    (data->'prediction'->>'confidence')::NUMERIC AS confidence,
    data->'prediction'->>'reasoning' AS reasoning,
    data->'prediction'->>'source' AS source,
    (data->'prediction'->>'safety_flag')::BOOLEAN AS safety_flag,
    (data->>'at')::TIMESTAMPTZ AS predicted_at
FROM public.civic_documents
WHERE collection = 'ai_predictions'
ORDER BY (data->>'at')::TIMESTAMPTZ DESC;

-- 5. Notifications Relational View
CREATE OR REPLACE VIEW public.notifications_view AS
SELECT
    id,
    data->>'user_id' AS user_id,
    data->>'type' AS notification_type,
    data->>'title' AS title,
    data->>'body' AS body,
    data->>'complaint_id' AS complaint_id,
    (data->>'read')::BOOLEAN AS is_read,
    (data->>'created_at')::TIMESTAMPTZ AS created_at
FROM public.civic_documents
WHERE collection = 'notifications'
ORDER BY (data->>'created_at')::TIMESTAMPTZ DESC;

-- 6. Analytics Reports Relational View
CREATE OR REPLACE VIEW public.analytics_reports_view AS
SELECT
    id,
    data->>'title' AS title,
    data->>'scope' AS scope,
    data->>'department' AS department,
    data->>'period_label' AS period_label,
    (data->'summary'->>'new_reports')::INTEGER AS new_reports,
    (data->'summary'->>'resolved')::INTEGER AS resolved,
    (data->'summary'->>'active')::INTEGER AS active,
    (data->'summary'->>'critical')::INTEGER AS critical,
    (data->>'generated_at')::TIMESTAMPTZ AS generated_at
FROM public.civic_documents
WHERE collection = 'analytics_reports'
ORDER BY (data->>'generated_at')::TIMESTAMPTZ DESC;

-- 7. Login & Registration Activity View
CREATE OR REPLACE VIEW public.login_activity_view AS
SELECT
    id,
    data->>'user_id' AS user_id,
    data->>'name' AS name,
    data->>'email' AS email,
    data->>'role' AS role,
    data->>'action' AS action,
    data->>'status' AS status,
    (data->>'timestamp')::TIMESTAMPTZ AS activity_at
FROM public.civic_documents
WHERE collection = 'login_activity'
ORDER BY (data->>'timestamp')::TIMESTAMPTZ DESC;

-- 8. Audit Logs View
CREATE OR REPLACE VIEW public.audit_logs_view AS
SELECT
    id,
    data->>'action' AS action,
    data->>'user_id' AS user_id,
    data->>'role' AS role,
    data->'meta' AS meta,
    (data->>'at')::TIMESTAMPTZ AS audit_at
FROM public.civic_documents
WHERE collection = 'audit_logs'
ORDER BY (data->>'at')::TIMESTAMPTZ DESC;
