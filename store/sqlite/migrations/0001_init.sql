-- Tables are the store's own; callers read the views below.

CREATE TABLE events (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT,
    parent_run_id TEXT,
    span_id       TEXT,
    kind          TEXT NOT NULL,
    step          INTEGER NOT NULL,
    time_ns       INTEGER NOT NULL,
    duration_ns   INTEGER NOT NULL,
    payload       TEXT NOT NULL
);

CREATE INDEX events_run ON events (run_id, id);
CREATE INDEX events_run_kind ON events (run_id, kind);
CREATE INDEX events_kind_time ON events (kind, time_ns);
CREATE INDEX events_parent ON events (parent_run_id) WHERE parent_run_id IS NOT NULL;

CREATE TABLE messages (
    hash TEXT PRIMARY KEY,
    body TEXT NOT NULL
) WITHOUT ROWID;

-- runs: one row per Start event, joined to its Finalize event when there is one.
-- status is NULL while a run has no Finalize (running, or the process died).
CREATE VIEW runs AS
SELECT s.run_id                                  AS run_id,
       s.parent_run_id                           AS parent_run_id,
       s.payload ->> '$.start.task'              AS task,
       s.time_ns                                 AS started_ns,
       f.time_ns                                 AS finished_ns,
       f.payload ->> '$.finalize.status'         AS status,
       f.payload ->> '$.finalize.truncation_reason' AS truncation_reason,
       f.step                                    AS steps,
       COALESCE(f.payload ->> '$.finalize.finalized', 0) AS finalized,
       f.payload ->> '$.finalize.final_text'     AS final_text,
       f.payload ->> '$.finalize.err'            AS err,
       COALESCE(f.payload ->> '$.finalize.usage.input_tokens', 0)  AS input_tokens,
       COALESCE(f.payload ->> '$.finalize.usage.output_tokens', 0) AS output_tokens
FROM events s
LEFT JOIN events f ON f.run_id = s.run_id AND f.kind = 'finalize'
WHERE s.kind = 'start';

-- spend: one row per Completion and per Decision event, the run's usage ledger.
CREATE VIEW spend AS
SELECT id                                        AS event_id,
       run_id                                    AS run_id,
       step                                      AS step,
       kind                                      AS kind,
       time_ns                                   AS time_ns,
       strftime('%Y-%m-%d', time_ns / 1000000000, 'unixepoch') AS day,
       COALESCE(payload ->> '$.completion.provider', payload ->> '$.decision.backend') AS provider,
       COALESCE(payload ->> '$.completion.model', payload ->> '$.decision.model')      AS model,
       payload ->> '$.completion.purpose'        AS purpose,
       COALESCE(payload ->> '$.completion.response.usage.input_tokens',
                payload ->> '$.decision.usage.input_tokens', 0)  AS input_tokens,
       COALESCE(payload ->> '$.completion.response.usage.output_tokens',
                payload ->> '$.decision.usage.output_tokens', 0) AS output_tokens,
       COALESCE(payload ->> '$.completion.response.usage.cache_read_input_tokens',
                payload ->> '$.decision.usage.cache_read_input_tokens', 0)     AS cache_read_tokens,
       COALESCE(payload ->> '$.completion.response.usage.cache_creation_input_tokens',
                payload ->> '$.decision.usage.cache_creation_input_tokens', 0) AS cache_creation_tokens,
       COALESCE(payload ->> '$.completion.response.usage.reasoning_tokens',
                payload ->> '$.decision.usage.reasoning_tokens', 0)            AS reasoning_tokens
FROM events
WHERE kind IN ('completion', 'decision');

-- tool_calls: one row per ToolRun event.
CREATE VIEW tool_calls AS
SELECT id                                          AS event_id,
       run_id                                      AS run_id,
       step                                        AS step,
       time_ns                                     AS time_ns,
       duration_ns                                 AS duration_ns,
       payload ->> '$.tool_run.call.id'            AS call_id,
       payload ->> '$.tool_run.call.name'          AS name,
       payload -> '$.tool_run.call.arguments'      AS arguments,
       payload ->> '$.tool_run.result'             AS result,
       COALESCE(payload ->> '$.tool_run.is_error', 0) AS is_error,
       COALESCE(payload ->> '$.tool_run.denied', 0)   AS denied,
       payload ->> '$.tool_run.deny_reason'        AS deny_reason
FROM events
WHERE kind = 'tool_run';

-- decisions: one row per answer of each Decision event; a call that returned
-- no answers (failed Ask) is one row with NULL answer columns and err set.
CREATE VIEW decisions AS
SELECT e.id                                  AS event_id,
       e.run_id                              AS run_id,
       e.step                                AS step,
       e.time_ns                             AS time_ns,
       e.payload ->> '$.decision.backend'    AS backend,
       e.payload ->> '$.decision.model'      AS model,
       e.payload ->> '$.decision.err'        AS err,
       a.value ->> '$.id'                    AS question_id,
       q.value ->> '$.kind'                  AS question_kind,
       a.value ->> '$.belief'                AS belief,
       a.value ->> '$.score'                 AS score,
       a.value ->> '$.choice'                AS choice,
       a.value ->> '$.confidence'            AS confidence
FROM events e
LEFT JOIN json_each(e.payload, '$.decision.answers') a
LEFT JOIN json_each(e.payload, '$.decision.questions') q ON q.value ->> '$.id' = a.value ->> '$.id'
WHERE e.kind = 'decision';
