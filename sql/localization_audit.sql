-- Fact-check requests carry language
SELECT TOP 100 claim_id, oracle_id, query_lang, queried_at
FROM dbo.fact_oracle_runs
ORDER BY queried_at DESC;

-- Gate language vs card language
SELECT TOP 100 g.detected_lang, r.target_lang, r.started, r.finished
FROM dbo.fact_gate_logs g
JOIN dbo.fact_runs r ON r.claim_id = g.msg_id
ORDER BY r.started DESC;

-- Runtime missing-key audit
SELECT *
FROM dbo.logs
WHERE event_type = 'i18n.missing_key'
ORDER BY created_at DESC;
