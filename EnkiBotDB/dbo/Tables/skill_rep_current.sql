CREATE TABLE [dbo].[skill_rep_current] (
    [chat_id] BIGINT        NOT NULL,
    [user_id] BIGINT        NOT NULL,
    [tag]     NVARCHAR (64) NOT NULL,
    [rep]     FLOAT (53)    NOT NULL,
    CONSTRAINT [PK_skill_rep_current] PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC, [tag] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_skill_tag]
    ON [dbo].[skill_rep_current]([tag] ASC, [rep] DESC);

