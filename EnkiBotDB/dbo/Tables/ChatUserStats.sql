CREATE TABLE [dbo].[ChatUserStats] (
    [ChatID]       BIGINT        NOT NULL,
    [UserID]       BIGINT        NOT NULL,
    [MessageCount] INT           DEFAULT ((0)) NOT NULL,
    [FirstSeen]    DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    [LastActive]   DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([ChatID] ASC, [UserID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ChatUserStats_ChatID_MessageCount]
    ON [dbo].[ChatUserStats]([ChatID] ASC, [MessageCount] DESC);

