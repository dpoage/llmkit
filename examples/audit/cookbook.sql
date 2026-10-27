-- Each query is one SELECT over the documented views (Q6 is the store's own
-- Source query over the events table). Parameters are named: :since_ns is
-- UTC Unix nanoseconds, :run a run id, :threshold a number.

-- Q1: spend per provider, model and UTC day since T. Index: events_kind_time.
SELECT day, provider, model,
       COUNT(*)                   AS calls,
       SUM(input_tokens)          AS input_tokens,
       SUM(output_tokens)         AS output_tokens,
       SUM(cache_read_tokens)     AS cache_read_tokens,
       SUM(cache_creation_tokens) AS cache_creation_tokens,
       SUM(reasoning_tokens)      AS reasoning_tokens
FROM spend
WHERE time_ns >= :since_ns
GROUP BY day, provider, model
ORDER BY day, provider, model;

-- Q2: the tool calls of a run in order. Index: events_run_kind.
SELECT step, name, arguments, arguments_b64, is_error, denied, deny_reason
FROM tool_calls
WHERE run_id = :run
ORDER BY event_id;

-- Q3: denied tool calls across runs since T. Index: events_kind_time.
SELECT time_ns, run_id, step, name, arguments, arguments_b64, deny_reason
FROM tool_calls
WHERE denied = 1 AND time_ns >= :since_ns
ORDER BY time_ns;

-- Q4: decisions whose score or belief is below a threshold since T. Index: events_kind_time.
SELECT time_ns, run_id, question_id, question_kind, score, belief
FROM decisions
WHERE time_ns >= :since_ns AND (score < :threshold OR belief < :threshold)
ORDER BY time_ns;

-- Q5: lineage of a run: its ancestors (events_run_kind) and descendants (events_parent).
-- CROSS JOIN keeps the walk driving and runs searched by index; a plain JOIN lets the planner scan.
WITH RECURSIVE
up(run_id, parent_run_id) AS (
    SELECT run_id, parent_run_id FROM runs WHERE run_id = :run
    UNION
    SELECT r.run_id, r.parent_run_id FROM up CROSS JOIN runs r ON r.run_id = up.parent_run_id
),
down(run_id, parent_run_id) AS (
    SELECT run_id, parent_run_id FROM runs WHERE parent_run_id = :run
    UNION
    SELECT r.run_id, r.parent_run_id FROM down CROSS JOIN runs r ON r.parent_run_id = down.run_id
)
SELECT 'ancestor' AS relation, run_id, parent_run_id FROM up WHERE run_id <> :run
UNION ALL
SELECT 'descendant', run_id, parent_run_id FROM down WHERE run_id <> :run;

-- Q6: a run's events in emission order. Index: events_run.
SELECT id, kind, payload
FROM events
WHERE run_id = :run
ORDER BY id;

-- Q7: runs that ended failed, panicked or incomplete since T. Index: events_kind_time.
SELECT run_id, parent_run_id, task, status, finished_ns, err
FROM runs
WHERE status IN ('failed', 'panicked', 'incomplete') AND finished_ns >= :since_ns
ORDER BY finished_ns;
