CREATE TABLE [dbo].[FactBookSources] (
    [id]           BIGINT          IDENTITY (1, 1) NOT NULL,
    [author]       NVARCHAR (256)  NOT NULL,
    [title]        NVARCHAR (512)  NOT NULL,
    [edition]      NVARCHAR (128)  NULL,
    [year]         INT             NULL,
    [isbn]         NVARCHAR (32)   NULL,
    [translator]   NVARCHAR (256)  NULL,
    [source_url]   NVARCHAR (1024) NULL,
    [snapshot_url] NVARCHAR (1024) NULL,
    [first_seen]   DATETIME2 (3)   DEFAULT (sysutcdatetime()) NOT NULL,
    PRIMARY KEY CLUSTERED ([id] ASC)
);

