CREATE TABLE [dbo].[UserUsage] (
    [UserID]     BIGINT NOT NULL,
    [UsageDate]  DATE   NOT NULL,
    [LlmCount]   INT    DEFAULT ((0)) NOT NULL,
    [ImageCount] INT    DEFAULT ((0)) NOT NULL,
    PRIMARY KEY CLUSTERED ([UserID] ASC, [UsageDate] ASC)
);

