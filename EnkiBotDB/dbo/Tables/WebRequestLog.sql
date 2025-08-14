CREATE TABLE [dbo].[WebRequestLog] (
    [LogID]      INT             IDENTITY (1, 1) NOT NULL,
    [Url]        NVARCHAR (1024) NOT NULL,
    [Method]     NVARCHAR (16)   NOT NULL,
    [StatusCode] INT             NULL,
    [DurationMs] INT             NULL,
    [Error]      NVARCHAR (512)  NULL,
    [Timestamp]  DATETIME2 (7)   DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_WebRequestLog_Timestamp]
    ON [dbo].[WebRequestLog]([Timestamp] DESC);

