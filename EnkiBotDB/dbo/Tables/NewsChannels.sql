CREATE TABLE [dbo].[NewsChannels] (
    [Username]  NVARCHAR (255) NOT NULL,
    [UpdatedAt] DATETIME2 (7)  DEFAULT (getdate()) NOT NULL,
    PRIMARY KEY CLUSTERED ([Username] ASC)
);

