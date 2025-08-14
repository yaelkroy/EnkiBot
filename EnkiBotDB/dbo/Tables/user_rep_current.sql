CREATE TABLE [dbo].[user_rep_current] (
    [chat_id]     BIGINT         NOT NULL,
    [user_id]     BIGINT         NOT NULL,
    [rep]         FLOAT (53)     DEFAULT ((0)) NOT NULL,
    [rep_global]  FLOAT (53)     DEFAULT ((0)) NOT NULL,
    [streak_days] INT            DEFAULT ((0)) NOT NULL,
    [last_seen]   DATETIME2 (3)  DEFAULT (sysutcdatetime()) NOT NULL,
    [badges_json] NVARCHAR (MAX) NULL,
    CONSTRAINT [PK_user_rep_current] PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_user_rep_rank]
    ON [dbo].[user_rep_current]([chat_id] ASC, [rep] DESC);

