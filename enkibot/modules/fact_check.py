# enkibot/modules/fact_check.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Minimal fact checking subsystem skeleton.

The real project described a production ready fact checking service.  This
module implements a *very* small portion of that design so the rest of the bot
can start integrating with it.  The goal of the skeleton is to provide the
same public interfaces as the full system so that future patches can increment
ally flesh out the behaviour.

The implementation here does not perform any network requests or heavy
processing – it merely wires together the classes, dataclasses and handler
structure described in the design document.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Message
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from ..utils.message_utils import get_text
from ..core.llm_services import LLMServices
import httpx
import logging
from ..utils.database import DatabaseManager

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .primary_source_hunter import PrimarySourceHunter, SourceHit

# Filter for messages that contain either plain text or a caption
TEXT_OR_CAPTION = (filters.TEXT & ~filters.COMMAND) | filters.CAPTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """Represents a normalised claim that can be checked."""

    text_norm: str
    text_orig: str
    lang: Optional[str]
    urls: List[str]
    hash: str


@dataclass
class Evidence:
    """Evidence item returned by a search fetcher."""

    url: str
    domain: str
    stance: str  # support|refute|mixed|na
    note: str
    published_at: Optional[str]
    snapshot_url: Optional[str]
    tier: Optional[int]
    score: float
    book: Optional["BookInfo"] = None


@dataclass
class Verdict:
    """Aggregated verdict for a claim."""

    label: str  # true|mostly_true|needs_context|unverified|false|misleading_media|opinion
    confidence: float
    summary: str
    sources: List[Evidence]


@dataclass
class SatireDecision:
    """Output of the satire detector."""

    p_meta: float
    p_text: float
    p_vis: float
    p_audio: float
    p_satire: float
    decision: str  # satire|ambiguous|news
    rationale: Dict[str, object]


@dataclass
class BookInfo:
    """Book metadata associated with an evidence item."""

    author: str
    title: str
    edition: Optional[str] = None
    year: Optional[int] = None
    isbn: Optional[str] = None
    page: Optional[str] = None
    chapter: Optional[str] = None
    translator: Optional[str] = None
    quote_exact: Optional[str] = None


@dataclass
class Quote:
    """Represents a book quote to verify."""

    quote: str
    author: Optional[str]
    title: Optional[str]
    lang: Optional[str]
    hash: str


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class Fetcher:
    """Interface for web fetchers.

    Real implementations should contact fact checking sites and general web
    search APIs.  The default implementation used here simply returns an empty
    list so the rest of the pipeline can continue to work without external
    services.
    """

    async def fact_checker_search(self, claim: Claim) -> List[Evidence]:
        return []

    async def general_search(self, claim: Claim) -> List[Evidence]:
        return []

    async def reverse_image(self, claim: Claim) -> List[Evidence]:
        return []


class DuckDuckGoFetcher(Fetcher):
    """Simple web fetcher using DuckDuckGo's public API."""

    async def fact_checker_search(self, claim: Claim) -> List[Evidence]:
        return await self.general_search(claim)

    async def general_search(self, claim: Claim) -> List[Evidence]:
        try:
            resp = await httpx.get(
                "https://r.jina.ai/http://api.duckduckgo.com/",
                params={"q": claim.text_norm, "format": "json", "no_redirect": "1", "no_html": "1"},
                timeout=10.0,
            )
            data = resp.json()
            evidences: List[Evidence] = []
            for topic in data.get("RelatedTopics", [])[:5]:
                url = topic.get("FirstURL")
                text = topic.get("Text", "")
                if not url:
                    continue
                domain = url.split("/")[2] if "//" in url else url
                stance = (
                    "refute"
                    if any(k in text.lower() for k in ["fake", "hoax", "debunk", "false"])
                    else "support"
                )
                evidences.append(
                    Evidence(
                        url=url,
                        domain=domain,
                        stance=stance,
                        note=text,
                        published_at=None,
                        snapshot_url=None,
                        tier=None,
                        score=1.0,
                    )
                )
            return evidences
        except Exception:
            return []


class StanceModel:
    """Assigns a stance/score to each evidence item."""

    async def classify(self, claim: Claim, evidences: List[Evidence]) -> List[Evidence]:
        return evidences


class SatireDetector:
    """Very small satire detector stub.

    The detector returns a constant ``news`` decision so it never blocks fact
    checking.  The interface mirrors the design document and can be extended
    later with real models.
    """

    def __init__(self, cfg_reader: Callable[[int], Dict[str, object]]):
        self.cfg_reader = cfg_reader

    async def predict(self, update: Update, text: str) -> SatireDecision:
        cfg = self.cfg_reader(update.effective_chat.id)
        weights = cfg.get("satire", {}).get(
            "weights", {"meta": 0.4, "text": 0.35, "vis": 0.2, "audio": 0.05}
        )
        p_meta = p_text = p_vis = p_audio = 0.0
        p_sat = 0.0
        return SatireDecision(
            p_meta=p_meta,
            p_text=p_text,
            p_vis=p_vis,
            p_audio=p_audio,
            p_satire=p_sat,
            decision="news",
            rationale={"features": {"meta": p_meta, "text": p_text, "vis": p_vis, "audio": p_audio}},
        )


class NewsGate:
    """Heuristic + classifier gate for detecting news-like text.

    The gate follows the revised high-recall strategy: a fast heuristic pass
    first, falling back to a lightweight keyword based classifier when the
    heuristics are inconclusive.  The goal is to catch genuine news forwards
    (including Russian/Ukrainian) without replying to non-news messages.
    """

    def __init__(self) -> None:
        # Heuristic feature lexicons -------------------------------------------------
        self.source_keywords = ["reuters", "ap", "bbc", "tass", "gov", "минобороны", "мчс"]
        self.news_verbs = [
            "said",
            "announced",
            "reported",
            "claims",
            "denied",
            "заявил",
            "сообщил",
            "отметил",
            "подчеркнул",
            "приведена",
            "признал",
        ]
        self.location_keywords = ["москва", "абхаз", "киев", "washington", "moscow"]
        self.crisis_keywords = ["эвакуац", "санкц", "землетряс", "атака", "боевую готовность"]
        self.time_re = re.compile(
            r"\b(\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|yesterday|"
            r"сегодня|вчера|понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)\b",
            re.I,
        )

        # Classifier keywords (used when heuristics are inconclusive) ---------------
        self.classifier_keywords = [
            *self.news_verbs,
            "according to",
            "today",
            "yesterday",
        ]

    # Heuristic pass ---------------------------------------------------------------
    def _heuristic(self, text_l: str) -> bool:
        if any(k in text_l for k in self.source_keywords):
            return True
        if any(v in text_l for v in self.news_verbs):
            return True
        if any(loc in text_l for loc in self.location_keywords):
            return True
        if any(c in text_l for c in self.crisis_keywords):
            return True
        if self.time_re.search(text_l):
            return True
        if re.search(r"https?://", text_l):
            return True
        return False

    # Lightweight classifier -------------------------------------------------------
    def _classifier(self, text_l: str) -> float:
        score = 0.0
        if any(k in text_l for k in self.classifier_keywords):
            score += 0.4
        if self.time_re.search(text_l):
            score += 0.2
        if re.search(r"https?://", text_l):
            score += 0.2
        if len(text_l) > 80:
            score += 0.2
        return min(score, 1.0)

    async def predict(self, text: str) -> float:
        text_l = text.lower()
        if self._heuristic(text_l):
            return 1.0
        return self._classifier(text_l)


class QuoteGate:
    """Simple detector for book-like quotations."""

    def __init__(self) -> None:
        self.quote_re = re.compile(r"[\"«].{8,}[\"»]")
        self.marker_keywords = [
            "как писал",
            "as",
            "wrote",
            "писал",
            "said",
            "\u2014",  # em dash
        ]

    async def predict(self, text: str) -> float:
        score = 0.0
        if self.quote_re.search(text):
            score += 0.5
        text_l = text.lower()
        if any(k in text_l for k in self.marker_keywords):
            score += 0.3
        if re.search(r"\b\d{4}\b", text_l):
            score += 0.1
        if "\n" in text:
            score += 0.1
        return min(score, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://\S+", re.I)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def hash_claim(text: str, urls: List[str]) -> str:
    canon = normalize_text(text).lower() + "\n" + "|".join(sorted(set(urls)))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fact checking orchestrator
# ---------------------------------------------------------------------------

class FactChecker:
    """Tiny orchestrator coordinating fact checks via an LLM."""

    def __init__(
        self,
        fetcher: Optional[Fetcher] = None,
        stance: Optional[StanceModel] = None,
        llm_services: Optional[LLMServices] = None,
        primary_hunter: Optional["PrimarySourceHunter"] = None,
    ):
        self.fetcher = fetcher
        self.stance = stance or StanceModel()
        self.llm_services = llm_services
        self.primary_hunter = primary_hunter

    async def extract_claim(self, text: str) -> Optional[Claim]:
        if not text or len(text) < 10:
            return None
        urls = URL_RE.findall(text)
        text_norm = normalize_text(text)
        return Claim(
            text_norm=text_norm,
            text_orig=text,
            lang=None,
            urls=urls,
            hash=hash_claim(text_norm, urls),
        )

    async def extract_quote(self, text: str) -> Optional[Quote]:
        """Extract a quotation and optional attribution."""
        m = re.search(r"[\"«](.+?)[\"»](?:\s*[\u2014\-]\s*([^\n]+))?", text, re.S)
        if not m:
            return None
        quote = normalize_text(m.group(1))
        author = title = None
        if m.group(2):
            parts = [p.strip() for p in re.split(r",|\u2014|-", m.group(2), maxsplit=1) if p.strip()]
            if parts:
                author = parts[0]
                if len(parts) > 1:
                    title = parts[1]
        h = hashlib.sha256(quote.lower().encode("utf-8")).hexdigest()
        return Quote(quote=quote, author=author, title=title, lang=None, hash=h)

    async def _llm_verdict(self, claim: Claim) -> Verdict:
        """Use LLM to generate a verdict for the claim."""
        if not self.llm_services:
            return Verdict(
                label="unverified",
                confidence=0.0,
                summary="LLM service unavailable.",
                sources=[],
            )

        system_prompt = (
            "You are a fact-checking assistant. Decide if the claim is true, false, "
            "needs_context, or unverified. If unsure, respond with unverified. "
            "Return a JSON object with keys 'label', 'confidence' (0-1), and 'summary'."
        )
        user_prompt = f"Claim: {claim.text_orig}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = await self.llm_services.call_openai_llm(
                messages, temperature=0.0, max_tokens=300
            )
        except Exception as e:
            logger.error(f"LLM fact-check failed: {e}", exc_info=True)
            completion = None

        label = "unverified"
        confidence = 0.0
        summary = "No analysis available."
        if completion:
            try:
                data = json.loads(completion)
                label = data.get("label", label)
                confidence = float(data.get("confidence", 0.0))
                summary = data.get("summary", summary)
            except Exception:
                summary = completion.strip()

        return Verdict(label=label, confidence=confidence, summary=summary, sources=[])

    async def _llm_quote_verdict(self, quote: Quote) -> Verdict:
        """Verify a book quotation using an LLM stub."""
        if not self.llm_services:
            return Verdict(
                label="unverified",
                confidence=0.0,
                summary="LLM service unavailable.",
                sources=[],
            )

        system_prompt = (
            "You verify literary quotations. Determine if the provided quote is "
            "accurate and correctly attributed. Respond with a JSON object "
            "containing keys 'label' (accurate|misquote|misattrib|unverified), "
            "'confidence' (0-1) and 'summary'."
        )
        user_prompt = f"Quote: \"{quote.quote}\"\nAuthor: {quote.author or 'unknown'}\nTitle: {quote.title or 'unknown'}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = await self.llm_services.call_openai_llm(
                messages, temperature=0.0, max_tokens=300
            )
        except Exception as e:
            logger.error(f"LLM quote-check failed: {e}", exc_info=True)
            completion = None

        label = "unverified"
        confidence = 0.0
        summary = "No analysis available."
        if completion:
            try:
                data = json.loads(completion)
                label = data.get("label", label)
                confidence = float(data.get("confidence", 0.0))
                summary = data.get("summary", summary)
            except Exception:
                summary = completion.strip()

        return Verdict(label=label, confidence=confidence, summary=summary, sources=[])

    async def research(self, claim: Claim | Quote, track: str = "news") -> Verdict:
        if track == "book":
            return await self._llm_quote_verdict(claim)  # type: ignore[arg-type]

        evidence: List[Evidence] = []
        if isinstance(claim, Claim) and self.primary_hunter:
            try:
                hits = await self.primary_hunter.hunt(claim.text_norm, claim.lang)
                for h in hits:
                    evidence.append(
                        Evidence(
                            url=h.url,
                            domain=h.domain,
                            stance="na",
                            note=h.title,
                            published_at=None,
                            snapshot_url=None,
                            tier=h.tier,
                            score=1.0,
                        )
                    )
                if evidence:
                    evidence = await self.stance.classify(claim, evidence)
            except Exception:
                evidence = []

        verdict = await self._llm_verdict(claim)  # type: ignore[arg-type]
        verdict.sources = evidence
        return verdict


# ---------------------------------------------------------------------------
# Telegram glue
# ---------------------------------------------------------------------------

class FactCheckBot:
    """Registers Telegram handlers for fact checking."""

    def __init__(
        self,
        app: Application,
        fc: FactChecker,
        satire_detector: Optional[SatireDetector] = None,
        news_gate: Optional[NewsGate] = None,
        quote_gate: Optional[QuoteGate] = None,
        cfg_reader: Callable[[int], Dict[str, object]] | None = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.app = app
        self.fc = fc
        self.satire = satire_detector or SatireDetector(lambda _chat_id: {})
        self.news_gate = news_gate or NewsGate()
        self.quote_gate = quote_gate or QuoteGate()
        self.cfg_reader = cfg_reader or (lambda _chat_id: {})
        self.db_manager = db_manager

    # Public API -------------------------------------------------------------
    def register(self) -> None:
        self.app.add_handler(CommandHandler("factcheck", self.cmd_factcheck))
        self.app.add_handler(
            MessageHandler(filters.FORWARDED & TEXT_OR_CAPTION, self.on_forward)
        )
        # Safety net for older PTB versions where Caption filter may not fire
        try:
            # PTB < v22
            document_filter = filters.DOCUMENT
        except AttributeError:
            # PTB v22+
            document_filter = filters.Document.ALL

        self.app.add_handler(
            MessageHandler(
                filters.FORWARDED & (filters.PHOTO | filters.VIDEO | document_filter),
                self.on_forward,
            )
        )
        self.app.add_handler(CallbackQueryHandler(self.on_factconfig_cb, pattern=r"^FC:"))
        self.app.add_handler(CommandHandler("factconfig", self.cmd_factconfig))

    # Handlers --------------------------------------------------------------
    async def on_forward(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = get_text(update.effective_message) or ""
        if not text and (
            update.effective_message.photo
            or update.effective_message.video
            or update.effective_message.document
        ):
            # Text-first workflow: only invoke OCR if the forward lacks text
            text = await self._ocr_extract(update.effective_message)
        cfg = self.cfg_reader(update.effective_chat.id)
        if cfg.get("satire", {}).get("enabled", True):
            dec = await self.satire.predict(update, text)
            await self._log_satire(update, dec)
            if dec.decision == "satire":
                kb = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Fact check anyway", callback_data="FC:FORCE")]]
                )
                await update.effective_message.reply_text(
                    "\ud83c\udccf Looks like satire/parody from this source.", reply_markup=kb
                )
                return
        if cfg.get("auto", {}).get("auto_check_news", True):
            p_news = await self.news_gate.predict(text)
            p_book = await self.quote_gate.predict(text)
            if self.db_manager and update.effective_chat:
                try:
                    await self.db_manager.log_fact_gate(
                        update.effective_chat.id,
                        update.effective_message.message_id,
                        p_news,
                        p_book,
                    )
                except Exception:
                    pass
            if p_book >= 0.70:
                await self._run_check(update, ctx, text, track="book")
                return
            if p_news >= 0.70:
                await self._run_check(update, ctx, text, track="news")
                return
            if p_book >= 0.55:
                await self._show_author_only_hint(update, ctx, "Check quote?", "book")
                return
            if p_news >= 0.55:
                await self._show_author_only_hint(update, ctx, "Check as news?", "news")
                return

    async def cmd_factcheck(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_message.reply_to_message:
            text = get_text(update.effective_message.reply_to_message) or ""
        else:
            text = " ".join(ctx.args)
        await self._run_check(update, ctx, text)

    async def _run_check(
        self,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        text: str,
        message: Optional[Message] = None,
        track: str = "news",
    ) -> None:
        """Run a fact check and react to the target message."""

        if track == "book":
            claim = await self.fc.extract_quote(text)
        else:
            claim = await self.fc.extract_claim(text)
        if not claim:
            return

        verdict = await self.fc.research(claim, track=track)

        target_msg = message or update.effective_message
        if self.db_manager and update.effective_chat and target_msg:
            try:
                await self.db_manager.log_fact_check(
                    update.effective_chat.id,
                    target_msg.message_id,
                    getattr(claim, "text_orig", getattr(claim, "quote", "")),
                    verdict.label,
                    verdict.confidence,
                    track,
                )
            except Exception as e:
                logger.error(f"Failed to log fact check: {e}", exc_info=True)

        try:
            if verdict.label in ("true", "mostly_true"):
                await target_msg.set_reaction("👍")
            else:
                await target_msg.set_reaction("👎")
                await target_msg.reply_text(
                    verdict.summary, disable_web_page_preview=True
                )
        except Exception:  # pragma: no cover - reaction support may vary
            if verdict.label not in ("true", "mostly_true"):
                await target_msg.reply_text(
                    verdict.summary, disable_web_page_preview=True
                )

    async def _ocr_extract(self, message: Message) -> str:
        """Extract text from media using a vision model (stub)."""
        # Placeholder: a real implementation would call GPT-4o or another OCR
        # service.  Returning an empty string keeps the pipeline silent when no
        # text is available.
        try:
            if self.fc.llm_services:
                # Actual OCR call would go here.
                pass
        except Exception:  # pragma: no cover - best effort
            pass
        return ""

    async def _show_author_only_hint(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, hint: str, track: str
    ) -> None:
        """Send a small hint to the original forwarder only."""
        kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton(hint, callback_data=f"FC:GATE:CHECK:{track}")]]
        )
        try:
            await ctx.bot.send_message(update.effective_user.id, hint, reply_markup=kb)
        except Exception:  # pragma: no cover - best effort
            pass

    def _format_card(self, v: Verdict) -> str:
        icon = {
            "true": "\u2705",
            "mostly_true": "\u2611\ufe0f",
            "needs_context": "\U0001f7e8",
            "unverified": "\ud83d\udd52",
            "false": "\u274c",
            "misleading_media": "\u26a0\ufe0f",
            "opinion": "\ud83d\udcac",
        }.get(v.label, "\u2139\ufe0f")
        lines = [
            f"{icon} Verdict: *{v.label.replace('_', ' ').title()}* ({v.confidence:.0%})",
            v.summary,
            "\nTop sources:",
        ]
        for e in v.sources:
            lines.append(
                f"\u2022 {e.domain} — {e.stance} {('('+e.published_at+')') if e.published_at else ''}"
            )
        return "\n".join(lines)

    # ---- /factconfig panel stubs -----------------------------------------
    async def cmd_factconfig(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Preset", callback_data="FC:TAB:Preset"),
                    InlineKeyboardButton("Sources", callback_data="FC:TAB:Sources"),
                ],
                [
                    InlineKeyboardButton("Policy", callback_data="FC:TAB:Policy"),
                    InlineKeyboardButton("Limits", callback_data="FC:TAB:Limits"),
                ],
                [
                    InlineKeyboardButton("Auto", callback_data="FC:TAB:Auto"),
                    InlineKeyboardButton("Danger", callback_data="FC:TAB:Danger"),
                ],
                [
                    InlineKeyboardButton("Export", callback_data="FC:EXPORT"),
                    InlineKeyboardButton("Apply", callback_data="FC:APPLY"),
                ],
            ]
        )
        await update.effective_message.reply_text("Fact check config:", reply_markup=kb)

    async def on_factconfig_cb(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        data = q.data
        await q.answer()
        if data == "FC:FORCE":
            if q.message and q.message.reply_to_message:
                orig = q.message.reply_to_message
                text = get_text(orig) or ""
                await q.edit_message_text("Checking…")
                await self._run_check(update, ctx, text, message=orig)
            else:
                await q.edit_message_text("Nothing to check.")
            return
        if data.startswith("FC:GATE:CHECK"):
            parts = data.split(":")
            track = parts[3] if len(parts) > 3 else "news"
            text = (q.message.text or "").replace(
                "\n\nCheck quote?", "").replace("\n\nCheck as news?", ""
            ).strip()
            await q.edit_message_text("Checking…")
            await self._run_check(update, ctx, text, track=track)
            return
        await q.edit_message_text("Config updated (stub).")

    # ------------------------------------------------------------------
    async def _log_satire(self, update: Update, dec: SatireDecision) -> None:
        """Persist satire decisions.

        Real implementation would insert a row into the SQL audit tables.  We
        simply log to console for now.
        """

        try:
            debug_data = json.dumps(dec.rationale)
        except Exception:  # pragma: no cover - best effort
            debug_data = "{}"
        update_str = f"chat={update.effective_chat.id} msg={update.effective_message.message_id}"
        print(f"Satire decision {dec.decision} for {update_str}: {debug_data}")
