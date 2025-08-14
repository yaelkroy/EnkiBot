CREATE TABLE [dbo].[trust_table] (
    [chat_id]        BIGINT        NOT NULL,
    [user_id]        BIGINT        NOT NULL,
    [trust]          FLOAT (53)    NOT NULL,
    [upheld]         INT           DEFAULT ((0)) NOT NULL,
    [overturned]     INT           DEFAULT ((0)) NOT NULL,
    [tenure_days]    INT           DEFAULT ((0)) NOT NULL,
    [phone_verified] BIT           DEFAULT ((0)) NOT NULL,
    [last_update]    DATETIME2 (3) DEFAULT (sysutcdatetime()) NOT NULL,
    CONSTRAINT [PK_trust_table] PRIMARY KEY CLUSTERED ([chat_id] ASC, [user_id] ASC)
);

