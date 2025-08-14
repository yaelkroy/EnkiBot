CREATE TABLE [dbo].[assistant_invocations] (
    [id]            BIGINT         IDENTITY (1, 1) NOT NULL,
    [chat_id]       BIGINT         NULL,
    [user_id]       BIGINT         NULL,
    [message_id]    BIGINT         NULL,
    [detected]      BIT            NULL,
    [alias]         NVARCHAR (32)  NULL,
    [prompt]        NVARCHAR (MAX) NULL,
    [reason]        NVARCHAR (64)  NULL,
    [lang]          NVARCHAR (8)   NULL,
    [routed_to_llm] BIT            NULL,
    [llm_ok]        BIT            NULL,
    [error]         NVARCHAR (512) NULL,
    [ts]            DATETIME2 (3)  DEFAULT (sysutcdatetime()) NOT NULL,
    PRIMARY KEY CLUSTERED ([id] ASC)
);

