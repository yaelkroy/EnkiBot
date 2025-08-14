CREATE TABLE [dbo].[UserNameVariations] (
    [VariationID]   INT            IDENTITY (1, 1) NOT NULL,
    [UserID]        BIGINT         NOT NULL,
    [NameVariation] NVARCHAR (255) NOT NULL,
    PRIMARY KEY CLUSTERED ([VariationID] ASC),
    FOREIGN KEY ([UserID]) REFERENCES [dbo].[UserProfiles] ([UserID]) ON DELETE CASCADE
);


GO
CREATE UNIQUE NONCLUSTERED INDEX [IX_UserNameVariations_NameVariation]
    ON [dbo].[UserNameVariations]([UserID] ASC, [NameVariation] ASC);


GO
CREATE UNIQUE NONCLUSTERED INDEX [IX_UserNameVariations_UserID_NameVariation]
    ON [dbo].[UserNameVariations]([UserID] ASC, [NameVariation] ASC);

