CREATE TABLE [dbo].[KarmaLog] (
    [LogID]          INT           IDENTITY (1, 1) NOT NULL,
    [ChatID]         BIGINT        NOT NULL,
    [GiverUserID]    BIGINT        NOT NULL,
    [ReceiverUserID] BIGINT        NOT NULL,
    [Points]         INT           NOT NULL,
    [Timestamp]      DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_KarmaLog_ChatID_Timestamp]
    ON [dbo].[KarmaLog]([ChatID] ASC, [Timestamp] DESC);

