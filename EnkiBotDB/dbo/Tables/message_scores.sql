CREATE TABLE [dbo].[message_scores] (
    [chat_id]       BIGINT        NOT NULL,
    [msg_id]        BIGINT        NOT NULL,
    [author_id]     BIGINT        NOT NULL,
    [score_current] FLOAT (53)    DEFAULT ((0)) NOT NULL,
    [pos]           INT           DEFAULT ((0)) NOT NULL,
    [neg]           INT           DEFAULT ((0)) NOT NULL,
    [last_update]   DATETIME2 (3) DEFAULT (sysutcdatetime()) NOT NULL,
    CONSTRAINT [PK_message_scores] PRIMARY KEY CLUSTERED ([chat_id] ASC, [msg_id] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_msg_author]
    ON [dbo].[message_scores]([author_id] ASC, [last_update] DESC);

