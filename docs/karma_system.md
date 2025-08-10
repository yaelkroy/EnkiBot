# EnkiBot Karma System Design

This document outlines the planned advanced karma system for EnkiBot. It is based on the "Better than Best Karma System" specification and captures the core ideas for implementation.

## Objectives
- **Fair** and skill oriented rather than pure popularity
- **Resilient** against brigading and Sybil attacks
- **Explainable** with transparent weight components
- **Real‑time** scoring for UX with batch decay for long term reputation

## User Experience
- Karma can be granted via emoji reactions or short phrases such as `+1`, `thanks`, or `спасибо`.
- Message score chips and leaderboards (weekly/monthly/seasonal) are supported.
- Personal `/karma` cards show global and per‑chat reputation.
- Downvotes require a quorum before applying.

## Scoring Model
```
weight = base × rater_trust × diversity × anti_collusion × novelty × content_factor
```
- `base` comes from the reaction type (e.g. helpful +1.0, insightful +1.2).
- `rater_trust` reflects account tenure and moderation history (0.5–1.5).
- `diversity` and `anti_collusion` damp repeated or reciprocal votes.
- `novelty` gives a short boost to early distinct raters.
- `content_factor` optionally rewards structured or long form content.

Message scores decay with `τ_msg` (default 7 days). User reputation is a decayed sum of message scores with `τ_user` (default 45 days).

## Data Model
Key tables introduced for the karma system:
- `karma_events` — append‑only ledger of vote events.
- `message_scores` — real‑time message aggregates.
- `user_rep_rollup` and `user_rep_current` — rolling reputation values.
- `skill_rep_current` — per‑topic expertise tracking.
- `karmaconfig` — per chat configuration (emoji map, decay, budgets, etc.).
- `trust_table` and `karmabans` — moderation helpers.

These tables are created automatically by `initialize_database()` and can be tuned further for partitioning or columnstore indexes in production deployments.

## Configuration & Admin Panel
An admin command `/karmaconfig` will expose presets (small/medium/large) and sliders for weights, decay, budgets, downvote quorum, and diversity thresholds. The configuration is stored in `karmaconfig` as JSON blobs and numeric fields, with server‑side validation to keep settings in safe ranges.

## Pipelines
- **Ingest** workers append events to `karma_events` and update `message_scores`.
- **Aggregator** tasks refresh message scores and leaderboard caches.
- **Daily rollups** update `user_rep_rollup` and `user_rep_current` applying decay.
- **Anti‑collusion** jobs analyse reciprocity and write dampening coefficients.

## Explainability
Vote tooltips show a breakdown of weight multipliers (e.g. `trust ×1.12, diversity ×0.88 → +0.99`). Users can query `/why @user` for recent reputation drivers.

## Security & Privacy
Only necessary IDs and timestamps are stored. Public views anonymize raters and full event export is restricted to admins. Users can export their own karma history via `/karmaexport`.

---
This document is an initial integration of the karma specification into the repository. The live code currently implements event logging and basic weight calculation; further modules (aggregation, admin UI, decay jobs) will build upon this foundation.
