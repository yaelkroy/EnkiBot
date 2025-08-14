CREATE TABLE [dbo].[FactCheckLog] (
    [LogID]      INT            IDENTITY (1, 1) NOT NULL,
    [ChatID]     BIGINT         NOT NULL,
    [MessageID]  BIGINT         NULL,
    [ClaimText]  NVARCHAR (MAX) NOT NULL,
    [Verdict]    NVARCHAR (50)  NOT NULL,
    [Confidence] FLOAT (53)     NOT NULL,
    [Timestamp]  DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    [Track]      NVARCHAR (8)   NULL,
    [Details]    NVARCHAR (MAX) NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC),
    CHECK ([Track]='book' OR [Track]='news')
);


GO
CREATE NONCLUSTERED INDEX [IX_FactCheckLog_ChatID_Timestamp]
    ON [dbo].[FactCheckLog]([ChatID] ASC, [Timestamp] DESC);

