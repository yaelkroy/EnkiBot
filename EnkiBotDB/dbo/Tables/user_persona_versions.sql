CREATE TABLE [dbo].[user_persona_versions] (
    [chat_id]      BIGINT         NOT NULL,
    [user_id]      BIGINT         NOT NULL,
    [version]      INT            NOT NULL,
    [created_at]   DATETIME2 (3)  DEFAULT (sysutcdatetime()) NOT NULL,
    [portrait_md]  NVARCHAR (MAX) NOT NULL,
    [traits_json]  NVARCHAR (MAX) NOT NULL,
    [signals_json] NVARCHAR (MAX) NULL,
    PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC, [version] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_user_persona_versions_lookup]
    ON [dbo].[user_persona_versions]([chat_id] ASC, [user_id] ASC, [version] DESC);

