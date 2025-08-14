CREATE TABLE [dbo].[ChatStats] (
    [ChatID]        BIGINT        NOT NULL,
    [TotalMessages] INT           DEFAULT ((0)) NOT NULL,
    [JoinCount]     INT           DEFAULT ((0)) NOT NULL,
    [LeaveCount]    INT           DEFAULT ((0)) NOT NULL,
    [LastUpdated]   DATETIME2 (7) DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([ChatID] ASC)
);

