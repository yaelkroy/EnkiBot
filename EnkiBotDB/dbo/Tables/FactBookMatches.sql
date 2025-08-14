CREATE TABLE [dbo].[FactBookMatches] (
    [match_id]       BIGINT         IDENTITY (1, 1) NOT NULL,
    [run_id]         BIGINT         NOT NULL,
    [book_source_id] BIGINT         NULL,
    [quote_exact]    NVARCHAR (MAX) NULL,
    [quote_lang]     NVARCHAR (8)   NULL,
    [page]           NVARCHAR (32)  NULL,
    [chapter]        NVARCHAR (64)  NULL,
    [stance]         NVARCHAR (12)  NULL,
    [score]          FLOAT (53)     NULL,
    PRIMARY KEY CLUSTERED ([match_id] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_FactBookMatches_Run]
    ON [dbo].[FactBookMatches]([run_id] ASC);

