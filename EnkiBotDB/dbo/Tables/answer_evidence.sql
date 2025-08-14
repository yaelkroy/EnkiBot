CREATE TABLE [dbo].[answer_evidence] (
    [answer_id]  BIGINT         IDENTITY (1, 1) NOT NULL,
    [chat_id]    BIGINT         NOT NULL,
    [asked_by]   BIGINT         NOT NULL,
    [intent]     NVARCHAR (32)  NOT NULL,
    [query_text] NVARCHAR (MAX) NULL,
    [lang]       NVARCHAR (8)   NULL,
    [created_at] DATETIME2 (3)  DEFAULT (sysutcdatetime()) NOT NULL,
    PRIMARY KEY CLUSTERED ([answer_id] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_answer_evidence_chat]
    ON [dbo].[answer_evidence]([chat_id] ASC, [created_at] DESC);

