# Memory and Portrait Tables

This document describes additional tables used to capture evidence for memory answers
and to version user psychological portraits.

## `answer_evidence`
Stores high-level information about evidence gathered for a bot answer.

```sql
CREATE TABLE answer_evidence (
  answer_id BIGINT IDENTITY PRIMARY KEY,
  chat_id   BIGINT NOT NULL,
  asked_by  BIGINT NOT NULL,
  intent    NVARCHAR(32) NOT NULL,
  query_text NVARCHAR(MAX) NULL,
  lang      NVARCHAR(8) NULL,
  created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
```

## `answer_evidence_items`
Links a particular answer to the messages that supported it.

```sql
CREATE TABLE answer_evidence_items (
  answer_id BIGINT NOT NULL,
  message_id BIGINT NOT NULL,
  rank INT NOT NULL,
  snippet NVARCHAR(MAX) NULL,
  reason NVARCHAR(64) NULL,
  PRIMARY KEY(answer_id, message_id)
);
```

## `user_persona_versions`
Maintains versioned Markdown portraits and traits for each user.

```sql
CREATE TABLE user_persona_versions (
  chat_id BIGINT NOT NULL,
  user_id BIGINT NOT NULL,
  version INT NOT NULL,
  created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
  portrait_md NVARCHAR(MAX) NOT NULL,
  traits_json NVARCHAR(MAX) NOT NULL,
  signals_json NVARCHAR(MAX) NULL,
  PRIMARY KEY(chat_id, user_id, version)
);
```
