CREATE TABLE [dbo].[ErrorLog] (
    [ErrorID]       INT            IDENTITY (1, 1) NOT NULL,
    [Timestamp]     DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    [LogLevel]      NVARCHAR (50)  NOT NULL,
    [LoggerName]    NVARCHAR (255) NULL,
    [ModuleName]    NVARCHAR (255) NULL,
    [FunctionName]  NVARCHAR (255) NULL,
    [LineNumber]    INT            NULL,
    [ErrorMessage]  NVARCHAR (MAX) NOT NULL,
    [ExceptionInfo] NVARCHAR (MAX) NULL,
    PRIMARY KEY CLUSTERED ([ErrorID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ErrorLog_Timestamp]
    ON [dbo].[ErrorLog]([Timestamp] DESC);

