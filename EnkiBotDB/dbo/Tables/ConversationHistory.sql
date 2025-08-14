CREATE TABLE [dbo].[ConversationHistory] (
    [MessageDBID] INT            IDENTITY (1, 1) NOT NULL,
    [ChatID]      BIGINT         NOT NULL,
    [UserID]      BIGINT         NOT NULL,
    [MessageID]   BIGINT         NULL,
    [Role]        NVARCHAR (50)  NOT NULL,
    [Content]     NVARCHAR (MAX) NOT NULL,
    [Timestamp]   DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([MessageDBID] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_ConversationHistory_ChatID_Timestamp]
    ON [dbo].[ConversationHistory]([ChatID] ASC, [Timestamp] DESC);

