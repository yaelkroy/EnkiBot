CREATE TABLE [dbo].[ChatLog] (
    [LogID]       INT            IDENTITY (1, 1) NOT NULL,
    [ChatID]      BIGINT         NOT NULL,
    [UserID]      BIGINT         NOT NULL,
    [Username]    NVARCHAR (255) NULL,
    [FirstName]   NVARCHAR (255) NULL,
    [MessageID]   BIGINT         NOT NULL,
    [MessageText] NVARCHAR (MAX) NULL,
    [Timestamp]   DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ChatLog_ChatID_Timestamp]
    ON [dbo].[ChatLog]([ChatID] ASC, [Timestamp] DESC);


GO
CREATE NONCLUSTERED INDEX [IX_ChatLog_UserID]
    ON [dbo].[ChatLog]([UserID] ASC);

