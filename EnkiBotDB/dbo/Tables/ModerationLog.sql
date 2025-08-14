CREATE TABLE [dbo].[ModerationLog] (
    [LogID]      INT            IDENTITY (1, 1) NOT NULL,
    [ChatID]     BIGINT         NOT NULL,
    [UserID]     BIGINT         NULL,
    [MessageID]  BIGINT         NOT NULL,
    [Categories] NVARCHAR (255) NULL,
    [Timestamp]  DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ModerationLog_ChatID_Timestamp]
    ON [dbo].[ModerationLog]([ChatID] ASC, [Timestamp] DESC);

