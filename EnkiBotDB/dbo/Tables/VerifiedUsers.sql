CREATE TABLE [dbo].[VerifiedUsers] (
    [UserID]     BIGINT        NOT NULL,
    [VerifiedAt] DATETIME2 (7) DEFAULT (getdate()) NULL,
    PRIMARY KEY CLUSTERED ([UserID] ASC)
);

