# Task for Codex — Comprehensive Localization Tests (EN & RU) + Auto-Fix Plan

> **Imperative policy**: Everything user-visible or user-bound must be localized. Globally and locally, every module, table, button, request, and answer must respect the user’s language setting. **Do not ship any English fallbacks in Russian flows (and vice versa).**

---

## 1) Objective

Deliver a repeatable test suite and fix pass that guarantees full localization coverage for English (en) and Russian (ru) across:

- **AI Assistant** (name triggers enki/энки/енки, replies, errors)
- **Fact-Checker** (news-only, envelopes, progress, book-quotes)
- **Karma** (votes, toasts, /karmaconfig)
- **Memory & Portraits** (/who_said, /persona, /memconfig)
- **Moderation** (/report, /spam, /voteban, /ban, /mute, /warn)
- **NSFW/Spam alerts**
- **Admin panels** (/langconfig, /factconfig, /karmaconfig, /memconfig)

Also validate that outbound requests include a language flag (e.g., `target_lang` for generation and `query_lang` for retrieval).

**Done means**: 0 missing i18n keys; 0 raw string literals in UI code; ≥99% language match rate in telemetry; pluralization correct per locale; CI green.

---

## 2) Pre-Flight (enforce policy)

1. Ensure the Localization Policy Header Pack is installed (headers, pre-commit, CI guard, SQL proc).
2. Verify base locale files: `i18n/en.json`, `i18n/ru.json` exist and load.
3. Adopt ICU MessageFormat for plural/gender where needed.
4. Confirm no hardcoded text: use an AST/lint pass to block literals in UI paths.

---

## 3) Static Checks (must pass before runtime tests)

### 3.1 Key coverage
Extract all keys referenced by code → compare against `en.json` and `ru.json`. Fail if any missing.

### 3.2 Literal string scan
Block user-visible string literals in UI and handler layers (allow exceptions for logs not shown to users).

### 3.3 SQL i18n properties
Run `dbo.usp_I18N_AssertProperties`; the result set must be empty (every text column/table storing user-visible text is tagged with `i18n:required=true`).

### 3.4 Pseudo-locale build
Add pseudo-locale `x-ps` that brackets and lengthens strings to catch truncation. Build must succeed with `LANG=x-ps`.

---

## 4) Runtime Tests (EN & RU)

For each module, test Happy path, Ambiguous, Error. Capture: input, detected lang, reply lang, buttons text, request `query_lang`, result, links/screenshot.

### 4.1 Language Detection & Routing
- RU prompt → RU reply; EN prompt → EN reply.
- RU caption + EN reply-question: response language follows the asker.
- Short tokens: «ок», «да», "ok" → if ambiguous, fallback message localized and chip buttons “🇷🇺/🇬🇧”.

### 4.2 AI Assistant
Triggers: `enki`, `энки`, `енки` (case-insensitive). Ensure replies, errors, and rate-limit messages localized.

### 4.3 Fact-Checker
- RU news forward → RU verdict card; `query_lang` includes `ru` and cross-lingual expansions.
- Non-news forward: silent; author-only hint localized.
- Book-quote: edition/page labels localized; titles can remain original in parentheses.
- Progress (if enabled): 1/3/6/10-min updates localized.

### 4.4 Karma
- Reactions → localized confirmations and toasts; pluralization:
  - ru: `{count, plural, one{# голос} few{# голоса} many{# голосов} other{# голосов}}`
  - en: `{count, plural, one{# vote} other{# votes}}`
- `/karmaconfig`: tabs, sliders, apply/export all localized.

### 4.5 Memory & Portraits
- `/who_said <query>`: RU in, RU out; quotes preserve original; surrounding text localized.
- `/persona @user`: portrait in RU/EN per request; disclaimers localized.

### 4.6 Moderation & Filters
- `/report`, `/spam`, `/voteban` → dialogs and confirmations localized.
- NSFW/Spam alerts localized.

### 4.7 Admin Panels
`/langconfig`, `/factconfig`, `/karmaconfig`, `/memconfig` complete localization: tabs, tooltips, diffs, errors.

---

## 5) Telemetry & Assertions

Log per interaction: `{detected_lang, target_lang, query_lang, card_lang}`.

Compute **Language Match Rate**: `card_lang == target_lang` must be ≥99% per day for RU & EN.

Verify `fact_oracle_runs.query_lang` includes RU for RU posts.

**SQL snippets**
```
-- Language match
SELECT CAST(SUM(CASE WHEN card_lang = target_lang THEN 1 ELSE 0 END) AS FLOAT)
       / NULLIF(COUNT(*),0) AS match_rate
FROM dbo.fact_runs
WHERE started >= DATEADD(day,-1,SYSUTCDATETIME());

-- Fact-check requests carry RU when needed
SELECT TOP 50 claim_id, query_lang, queried_at
FROM dbo.fact_oracle_runs
WHERE query_lang LIKE '%ru%'
ORDER BY queried_at DESC;
```

---

## 6) Auto-Fix Playbook (what to do when tests fail)

1. Missing key → add to `en.json` and `ru.json`. Prefer ICU forms for plurals. Commit with test.
2. Literal detected → replace with `t('path.key')`. Add a unit test that fails if code regresses.
3. Wrong language reply → enforce `target_lang` in the system prompt; add post-generation language check (detect → regenerate/translate); write an integration test.
4. Buttons unlocalized → move inline texts to i18n keys; verify in snapshot test.
5. Bad pluralization → migrate to ICU; add count-based tests (1, 2–4, 5–20, 21).
6. Requests missing `query_lang` → plumb language through the request builder; log and assert presence in CI.
7. Truncation in RU → shorten text or widen UI; confirm with pseudo-locale.

Each fix must include: locale updates, unit/snapshot test, and a screenshot for RU & EN.

---

## 7) Automation Targets

- **Unit tests**: language selector, ICU plural helper, request builder (ensures `query_lang`).
- **Integration tests**: end-to-end for AI assistant, fact-checker, karma, `/persona`.
- **Snapshot tests**: cards/buttons for RU & EN (plus pseudo-locale).
- **Lint**: block string literals.
- **CI gates**: fail on missing keys or SQL i18n properties.

---

## 8) Deliverables

1. **PR #1 — Test scaffolding**: unit/integration/snapshot tests; pseudo-locale; CI hooks.
2. **PR #2 — Fixes**: locale fill-ins, plural ICU forms, code wiring for `target_lang`/`query_lang`.
3. **PR #3 — SQL i18n tags**: extended properties on all relevant tables/columns.
4. **Report**:
   - Test matrix results with screenshots (EN & RU)
   - Language match rate ≥99% proof
   - SQL audit outputs (zero missing props, correct `query_lang`)

---

## 9) Acceptance Criteria (hard)

- No missing i18n keys for EN/RU.
- Zero user-visible literals in UI code.
- Language match rate ≥99% (daily) for both languages.
- Pluralization correct on 1/2/5/21 counts in RU; EN plurals correct.
- All buttons, replies, hints, progress, errors localized.
- All outbound requests carry `query_lang`; LLM prompts carry `target_lang`.

---

## 10) Notes for Future Developers

- Always add strings to i18n files first; never inline.
- Prefer ICU for plurals/gender/case.
- RU text is often longer; design for 20–35% expansion.
- Keep English outlet/source names; localize surrounding labels.
- Re-run the full suite whenever adding modules/features.

