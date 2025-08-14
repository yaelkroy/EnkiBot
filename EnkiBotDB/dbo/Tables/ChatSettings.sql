CREATE TABLE [dbo].[ChatSettings] (
    [ChatID]            BIGINT     NOT NULL,
    [SpamVoteThreshold] INT        DEFAULT ((3)) NOT NULL,
    [NSFWFilterEnabled] BIT        DEFAULT ((0)) NOT NULL,
    [NSFWThreshold]     FLOAT (53) DEFAULT ((0.8)) NOT NULL,
    PRIMARY KEY CLUSTERED ([ChatID] ASC)
);

