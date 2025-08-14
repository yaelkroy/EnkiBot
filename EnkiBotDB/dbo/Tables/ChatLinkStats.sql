CREATE TABLE [dbo].[ChatLinkStats] (
    [ChatID]    BIGINT         NOT NULL,
    [Domain]    NVARCHAR (255) NOT NULL,
    [LinkCount] INT            DEFAULT ((0)) NOT NULL,
    PRIMARY KEY CLUSTERED ([ChatID] ASC, [Domain] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ChatLinkStats_ChatID_Count]
    ON [dbo].[ChatLinkStats]([ChatID] ASC, [LinkCount] DESC);

