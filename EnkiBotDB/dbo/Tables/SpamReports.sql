CREATE TABLE [dbo].[SpamReports] (
    [ReportID]       INT           IDENTITY (1, 1) NOT NULL,
    [ChatID]         BIGINT        NOT NULL,
    [TargetUserID]   BIGINT        NOT NULL,
    [ReporterUserID] BIGINT        NOT NULL,
    [Timestamp]      DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([ReportID] ASC),
    CONSTRAINT [UQ_SpamReports] UNIQUE NONCLUSTERED ([ChatID] ASC, [TargetUserID] ASC, [ReporterUserID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_SpamReports_Chat_Target]
    ON [dbo].[SpamReports]([ChatID] ASC, [TargetUserID] ASC);

