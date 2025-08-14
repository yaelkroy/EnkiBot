CREATE TABLE [dbo].[FactGateLog] (
    [LogID]     INT           IDENTITY (1, 1) NOT NULL,
    [ChatID]    BIGINT        NOT NULL,
    [MessageID] BIGINT        NOT NULL,
    [PNews]     FLOAT (53)    NULL,
    [PBook]     FLOAT (53)    NULL,
    [Timestamp] DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([LogID] ASC)
);

