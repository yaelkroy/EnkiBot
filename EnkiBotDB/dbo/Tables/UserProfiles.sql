CREATE TABLE [dbo].[UserProfiles] (
    [UserID]             BIGINT         NOT NULL,
    [Username]           NVARCHAR (255) NULL,
    [FirstName]          NVARCHAR (255) NULL,
    [LastName]           NVARCHAR (255) NULL,
    [LastSeen]           DATETIME2 (7)  DEFAULT (getdate()) NULL,
    [MessageCount]       INT            DEFAULT ((0)) NULL,
    [PreferredLanguage]  NVARCHAR (10)  NULL,
    [Notes]              NVARCHAR (MAX) NULL,
    [ProfileLastUpdated] DATETIME2 (7)  DEFAULT (getdate()) NULL,
    [KarmaReceived]      INT            DEFAULT ((0)) NOT NULL,
    [KarmaGiven]         INT            DEFAULT ((0)) NOT NULL,
    [HateGiven]          INT            DEFAULT ((0)) NOT NULL,
    PRIMARY KEY CLUSTERED ([UserID] ASC)
);

