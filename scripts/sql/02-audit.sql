-- Append-only audit log. One row per turn, PII redacted.
--
-- Phase 9 will replace this with a DynamoDB on-demand table; the write
-- shape stays the same.

CREATE TABLE IF NOT EXISTS audit_events (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id               TEXT NOT NULL,
    conversation_id          TEXT NOT NULL,
    ts                       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- conversation snapshot
    user_message_redacted    TEXT NOT NULL,
    response_redacted        TEXT,
    intent                   TEXT,
    user_type                TEXT,
    confidence               REAL,
    route                    TEXT,

    -- LLM accounting
    input_tokens             INT  NOT NULL DEFAULT 0,
    output_tokens            INT  NOT NULL DEFAULT 0,
    cost_usd                 NUMERIC(10, 6) NOT NULL DEFAULT 0,
    latency_ms               INT  NOT NULL DEFAULT 0,

    -- diagnostics
    error                    TEXT,
    metadata                 JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS audit_events_conv_idx ON audit_events(conversation_id);
CREATE INDEX IF NOT EXISTS audit_events_ts_idx ON audit_events(ts DESC);
CREATE INDEX IF NOT EXISTS audit_events_intent_idx ON audit_events(intent);
