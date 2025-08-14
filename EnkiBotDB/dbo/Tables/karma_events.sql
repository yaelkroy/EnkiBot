CREATE TABLE [dbo].[karma_events] (
    [event_id]       BIGINT        IDENTITY (1, 1) NOT NULL,
    [chat_id]        BIGINT        NOT NULL,
    [msg_id]         BIGINT        NULL,
    [target_user_id] BIGINT        NOT NULL,
    [rater_user_id]  BIGINT        NOT NULL,
    [emoji]          NVARCHAR (16) NULL,
    [base]           FLOAT (53)    NOT NULL,
    [rater_trust]    FLOAT (53)    NOT NULL,
    [diversity]      FLOAT (53)    NOT NULL,
    [anti_collusion] FLOAT (53)    NOT NULL,
    [novelty]        FLOAT (53)    NOT NULL,
    [content_factor] FLOAT (53)    NOT NULL,
    [weight]         FLOAT (53)    NOT NULL,
    [ts]             DATETIME2 (3) DEFAULT (sysutcdatetime()) NOT NULL,
    PRIMARY KEY CLUSTERED ([event_id] ASC)
);


GO
CREATE NONCLUSTERED INDEX [IX_events_chat_msg]
    ON [dbo].[karma_events]([chat_id] ASC, [msg_id] ASC)
    INCLUDE([ts], [weight]);

