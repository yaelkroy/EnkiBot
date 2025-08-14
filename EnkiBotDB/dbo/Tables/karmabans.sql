CREATE TABLE [dbo].[karmabans] (
    [chat_id]  BIGINT         NOT NULL,
    [user_id]  BIGINT         NOT NULL,
    [reason]   NVARCHAR (256) NULL,
    [until_ts] DATETIME2 (7)  NULL,
    CONSTRAINT [PK_karmabans] PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC)
);

