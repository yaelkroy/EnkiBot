CREATE TABLE [dbo].[user_rep_rollup] (
    [chat_id]     BIGINT         NOT NULL,
    [user_id]     BIGINT         NOT NULL,
    [day]         DATE           NOT NULL,
    [delta_score] FLOAT (53)     NOT NULL,
    [pos]         INT            NOT NULL,
    [neg]         INT            NOT NULL,
    [skills_json] NVARCHAR (MAX) NULL,
    CONSTRAINT [PK_user_rep_rollup] PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC, [day] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_user_rep_rollup_day]
    ON [dbo].[user_rep_rollup]([day] ASC);

