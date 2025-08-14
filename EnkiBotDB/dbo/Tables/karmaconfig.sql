CREATE TABLE [dbo].[karmaconfig] (
    [chat_id]                BIGINT         NOT NULL,
    [emoji_map_json]         NVARCHAR (MAX) NULL,
    [decay_msg_days]         INT            DEFAULT ((7)) NOT NULL,
    [decay_user_days]        INT            DEFAULT ((45)) NOT NULL,
    [allow_downvotes]        BIT            DEFAULT ((1)) NOT NULL,
    [daily_budget]           INT            DEFAULT ((18)) NOT NULL,
    [downvote_quorum]        INT            DEFAULT ((4)) NOT NULL,
    [diversity_window_hours] INT            DEFAULT ((12)) NOT NULL,
    [reciprocity_threshold]  FLOAT (53)     DEFAULT ((0.30)) NOT NULL,
    [preset]                 NVARCHAR (32)  NULL,
    [auto_tune]              BIT            DEFAULT ((1)) NOT NULL,
    PRIMARY KEY CLUSTERED ([chat_id] ASC)
);

