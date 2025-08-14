CREATE TABLE [dbo].[answer_evidence_items] (
    [answer_id]  BIGINT         NOT NULL,
    [message_id] BIGINT         NOT NULL,
    [rank]       INT            NOT NULL,
    [snippet]    NVARCHAR (MAX) NULL,
    [reason]     NVARCHAR (64)  NULL,
    PRIMARY KEY CLUSTERED ([answer_id] ASC, [message_id] ASC)
);

