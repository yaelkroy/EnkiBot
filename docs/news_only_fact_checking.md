# News‑Only Fact‑Checking — Oracle‑Based Deep Research (o3/o4‑mini + GPT‑4o)

This document defines a **best‑of‑the‑best** mechanism for fact‑checking **forwarded news** in Telegram chats. The bot itself **never decides truth**. It orchestrates deep‑search **oracles** and reports their evaluations, with transparent sources and confidence.

> **Model routing rules** (built‑in):
>
> * **Best quality:** `o3-deep-research` → embeddings → web tools/oracles → **reported verdict**
> * **Balanced cost/latency:** `o4-mini-deep-research` → embeddings → web tools/oracles
> * **Lots of images:** `GPT-4o` for extraction (OCR/Vision) → `o3`/`o4-mini` for synthesis, still **citing oracles**

---

## 0) Trust Boundary

* **News‑only:** The pipeline first decides whether a forward is *news*. Non‑news (jokes, ads, memes, personal notes) are ignored.
* **Oracle‑based:** Truth assessments come from **external providers** (search/fact‑check/news APIs). The bot/LLM may *synthesize* but **must not invent** a verdict; it **reports** provider findings and consensus/disagreement.
* **Explainable:** Every card shows provider names, timestamps, links/snapshots, and confidence *as reported by providers*.

---

## 1) Triggers & Guards

**When to run**

* Forwarded messages (`forward_from`/`forward_from_chat`) with text, URL, or media caption.
* Manual: reply `/factcheck` or 🧪 reaction to a message.

**Guards**

* Per‑chat daily cap; burst queue.
* Admin setting `forwards_only | forwards_high_risk | off`.

---

## 2) Newsness Gate (only news proceeds)

A two‑stage gate: cheap heuristics first, then a lightweight classifier. High recall; multilingual.

* **Heuristic pass** (accept if any matches): source signals, headline+lede pattern, event time/place, quotative verbs, crisis lexicon, or article screenshot cues.
* **Classifier pass**: tiny multilingual model trained on news vs. commentary/opinion/jokes.
* **Policy**: `P_news ≥ 0.70` → proceed; `0.55–0.70` → *hold* (silent) and show the author a private **“Check as news?”** button; `<0.55` → *reject* silently.

---

## 3) Silent UX Policy

* **Non‑news**: the bot posts nothing. Optional admin digests only.
* **Ambiguous**: only the author receives a hint with **“Check as news?”**. Group stays quiet.
* **News**: after research, a normal verdict card appears.

---

## 4) Claim Extraction & Normalization

1. Canonicalize text; collect URLs; language‑detect; translate for research; keep original.
2. For images/videos: **GPT‑4o** extracts on‑screen text, captions, speaker quotes, and watermarks; grab keyframes.
3. Decompose into **atomic claims** (subject • predicate • object • where/when • quantity). Compute `claim_hash = SHA256(norm+entities+date_hint)` to dedupe.

---

## 5) Evidence Oracles (Deep Search)

**Providers (examples; pluggable):**

* **Fact‑check aggregators:** IFCN members (AP/Reuters/AFP fact checks, PolitiFact, FullFact, Snopes, HealthFeedback, etc.).
* **Structured news APIs:** event‑centric feeds (e.g., Perigon, GDELT, NewsAPI‑like) with outlet metadata.
* **General web search:** metasearch APIs for coverage diversity.
* **Primary sources:** official portals (gov/regulator/WHO/CDC press rooms).
* **Media verification:** reverse‑image (Lens/TinEye/Bing Visual), YouTube frame search, Wayback/CDX snapshots.

**Normalization contract (per provider response)**

```
{
  "provider": "string",
  "verdict": "support|refute|mixed|context|unknown|opinion",
  "confidence": 0.0,
  "claim_text": "string",
  "evidence": [
    {"url":"https://…","domain":"…","title":"…","published_at":"…",
     "stance":"support|refute|mixed|na","note":"…","snapshot":"https://…",
     "tier":1,"score":0.73}
  ]
}
```

---

## 6) Model Orchestration (Responses API)

**Why use the model?** To plan, call tools, cluster duplicates, align provider labels, and write a careful, neutral summary. **Not** to invent truth.

**Pipelines**

* **Premium:** `o3-deep-research` + embeddings (candidate retrieval) + oracle tools → *reported verdict*.
* **Balanced:** `o4-mini-deep-research` with same toolset.
* **Vision‑heavy:** `GPT-4o` (extraction) → `o3/o4-mini` for planning/synthesis.

**Tool schema (examples)**

```
web.search: {"q":"string","recencyDays":30,"lang":"auto"}
web.open:   {"url":"string"}
news.lookup: {"query":"string","from":"ISO date","to":"ISO date"}
factcheck.lookup: {"claim":"string"}
archive.snapshot: {"url":"string"}
```

**System prompt key rules**

* “You are an *orchestrator*. Gather provider evaluations; do **not** generate your own verdict. Map provider labels to UI labels without changing meaning. If providers disagree, surface disagreement.”

---

## 7) Consensus & Disagreement

* **Consensus**: ≥2 independent providers report the same stance → show *Confirmed* or *Contradicted* with provider list; confidence = **max provider confidence** (never invented).
* **Split**: providers disagree → show counts (e.g., `2 support / 1 refute`) and a neutral summary. Offer **“More sources”**.
* **Unverified/Developing**: no credible coverage yet → show *Unverified* + re‑check schedule.

---

## 8) Verdict Card (Telegram UI)

* Header: label + emoji (✅/❌/🟨/🕒/⚠️/💬)
* One‑line neutral reason (e.g., “Reuters & AP say the claim is incorrect; see sources”).
* **Provider badges** with confidence, then 3–6 evidence links (tap opens).
* Buttons: **Evidence • Timeline • Explain • Disagree?**
* Edits on updates (“Updated 12:04 UTC”).

---

## 9) Embeddings & Speed‑ups

* Build an **embeddings** index of article titles/snippets and past claims (e.g., `text-embedding-3-large`).
* Use it to: (1) rank URLs for opening, (2) detect duplicates, (3) pre‑seed `factcheck.lookup` queries.
* Cache results for `claim_hash` to answer repeated forwards instantly.

---

## 10) News‑Only Flow (end‑to‑end)

1. **Forward arrives** → Satire/Jokes filter → **Newsness Gate**.
2. If news: extract claims; compute `claim_hash`; lookup cache.
3. Select **route**: `o3` (quality) or `o4-mini` (balanced); if images, add **GPT‑4o** step.
4. Model plans tool calls → hits **oracles** → gathers provider responses.
5. Normalize labels → compute **consensus/split** (display‑only) → store raw JSON & snapshots.
6. Render verdict card; schedule re‑check for *Unverified/Developing*.

---

## 11) Admin Controls (`/factconfig` excerpts)

* **Auto‑trigger**: `forwards_only | forwards_high_risk | off`
* **Min provider tier**: 1/2
* **Max latency**: e.g., 15 s (then partial results)
* **Re‑check cadence**: developing stories (2h), unverified (6h)
* **Model policy**: `o3|o4-mini|auto` and **Vision assist** on/off

---

## 12) Storage (SQL quick sketch)

* `fact_oracle_runs(claim_id, oracle_id, verdict, confidence, queried_at, raw_json)`
* `fact_oracle_evidence(run_id, url, domain, title, stance, published_at, snapshot_url, score)`
* Index on `(claim_id, queried_at desc)`

---

## 13) Safety & Compliance

* Never assert truth beyond providers; attribute clearly.
* Defamation guard: stricter policy for allegations about private individuals.
* Regional/legal: respect provider ToS; keep snapshots for audit.

---

## 14) Quality & Monitoring

* **Coverage**: share of news forwards that pass the Gate and receive provider data.
* **Disagreement rate** and time to consensus.
* **Latency P50/P95** end‑to‑end.
* **Appeal outcomes** (when users submit counter‑evidence).

---

## 15) Minimal Orchestrator Pseudocode

```python
async def handle_forward(msg):
    if not news_gate(msg):
        return  # silent for non-news
    claim = extract_claim(msg)
    if cache.has(claim.hash): return render(cache.get(claim.hash))
    route = choose_route(msg)   # o3 | o4-mini (+ 4o if images)
    results = await run_oracles(route, claim)
    ui = normalize_and_render(results)   # consensus or split (no invented verdict)
    cache.set(claim.hash, ui)
    return ui
```

---

### Takeaway

This design **checks only news**, delegates truth to **deep‑search oracles**, and uses OpenAI models purely as **planners/synthesizers** under strict attribution. Pick the **o3** route for maximum quality, **o4‑mini** for cost/latency, and add **GPT‑4o** whenever images are involved.

