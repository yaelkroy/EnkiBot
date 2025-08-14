# enkibot/modules/fact_check.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal fact checking subsystem skeleton.

This module provides a compact, working subset of the fact-checking
infrastructure so the rest of the bot can integrate with stable public
interfaces. It includes:
- Data models (Claim, Evidence, Verdict, Quote, etc.)
- Lightweight gates (news/quote) with debug reasons
- A skeleton web fetcher (optional OpenAI web_search)
- An LLM-driven verdict generator (descriptive on errors)
- Telegram glue (forwarded message handler, commands, callbacks)

Network calls are best-effort and optional; the system stays silent when
unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse
from io import BytesIO

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Message
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import create_deep_linked_url
from telegram.error import BadRequest

from ..utils.message_utils import get_text, is_forwarded_message
from ..core.llm_services import LLMServices
from .. import config
from ..utils.database import DatabaseManager
from ..utils.lang_router import normalize as normalize_unicode

from types import SimpleNamespace
try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover
    openai = SimpleNamespace()

if TYPE_CHECKING:  # pragma: no cover - hints only
    from .primary_source_hunter import PrimarySourceHunter

# Filter for messages that contain either text or a caption (non-commands)
TEXT_OR_CAPTION = (filters.TEXT & ~filters.COMMAND) | filters.CAPTION

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency for language detection
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    detect = None  # type: ignore
    LangDetectException = Exception  # type: ignore

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    text_norm: str
    text_orig: str
    lang: Optional[str]
    urls: List[str]
    hash: str


@dataclass
class Evidence:
    url: str
    domain: str
    stance: str  # support|refute|mixed|na
    note: str
    published_at: Optional[str]
    snapshot_url: Optional[str]
    tier: Optional[int]
    score: float


@dataclass
class Verdict:
    label: str  # true|mostly_true|needs_context|unverified|false|misleading_media|opinion
    confidence: float
    summary: str
    sources: List[Evidence]
    debug: List[str] = field(default_factory=list)


@dataclass
class SatireDecision:
    p_meta: float
    p_text: float
    p_vis: float
    p_audio: float
    p_satire: float
    decision: str  # satire|ambiguous|news
    rationale: Dict[str, object]


@dataclass
class BookInfo:
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
    quote: str
    author: Optional[str]
    title: Optional[str]
    lang: Optional[str]
    hash: str


# ---------------------------------------------------------------------------
# Gates and helpers
# ---------------------------------------------------------------------------

class SatireDetector:
    """Very small satire detector stub always returning 'news'."""

    def __init__(self, cfg_reader: Callable[[int], Dict[str, object]]):
        self.cfg_reader = cfg_reader

    async def predict(self, update: Update, text: str) -> SatireDecision:
        return SatireDecision(
            p_meta=0.0,
            p_text=0.0,
            p_vis=0.0,
            p_audio=0.0,
            p_satire=0.0,
            decision="news",
            rationale={"features": {}},
        )


class NewsGate:
    """Heuristic + classifier gate for news-like text with reasons."""

    def __init__(self) -> None:
        self.source_keywords = ["reuters", "ap", "bbc", "tass", "gov", "минобороны", "мчс"]
        self.news_verbs = [
            "said", "announced", "reported", "claims", "denied",
            "заявил", "заявила", "заявили",
            "сообщил", "сообщила", "сообщили", "сообщают",
            "отметил", "подчеркнул", "приведена", "признал",
            "подтвердил", "подтвердили",
        ]
        self.location_keywords = ["москва", "абхаз", "киев", "washington", "moscow"]
        self.crisis_keywords = ["эвакуац", "санкц", "землетряс", "атака", "боевую готовность"]
        # English months + Russian months (stems to match cases: января/январе, августа/августе, etc.)
        self.time_re = re.compile(
            r"\b(\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|yesterday|"
            r"сегодня|вчера|понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье|"
            r"январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йея]|июн[ьяе]|июл[ьяе]|август[ане]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])\b",
            re.I,
        )
        self.classifier_keywords = [*self.news_verbs, "according to", "today", "yesterday"]

    def _heuristic(self, text_l: str) -> bool:
        return (
            any(k in text_l for k in self.source_keywords)
            or any(v in text_l for v in self.news_verbs)
            or any(loc in text_l for loc in self.location_keywords)
            or any(c in text_l for c in self.crisis_keywords)
            or bool(self.time_re.search(text_l))
            or bool(re.search(r"https?://", text_l))
        )

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
        return 1.0 if self._heuristic(text_l) else self._classifier(text_l)

    def debug_predict(self, text: str) -> tuple[float, List[str]]:
        text_l = text.lower()
        reasons: List[str] = []
        if any(k in text_l for k in self.source_keywords):
            reasons.append("heuristic:source")
        if any(v in text_l for v in self.news_verbs):
            reasons.append("heuristic:verb")
        if any(loc in text_l for loc in self.location_keywords):
            reasons.append("heuristic:location")
        if any(c in text_l for c in self.crisis_keywords):
            reasons.append("heuristic:crisis")
        if self.time_re.search(text_l):
            reasons.append("heuristic:time")
        if re.search(r"https?://", text_l):
            reasons.append("heuristic:url")
        if reasons:
            return 1.0, reasons
        score = 0.0
        if any(k in text_l for k in self.classifier_keywords):
            score += 0.4; reasons.append("classifier:keywords")
        if self.time_re.search(text_l):
            score += 0.2; reasons.append("classifier:time")
        if re.search(r"https?://", text_l):
            score += 0.2; reasons.append("classifier:url")
        if len(text_l) > 80:
            score += 0.2; reasons.append("classifier:length>80")
        return min(score, 1.0), reasons


class QuoteGate:
    def __init__(self) -> None:
        # Require two or more words inside quotes
        self.quote_re = re.compile(r"[\"«](?:(?![\"»]).)*\s+(?:(?![\"»]).)+[\"»]")
        self.marker_keywords = ["как писал", "as", "wrote", "писал", "said", "\u2014"]

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
    s = normalize_unicode(s)
    return re.sub(r"\s+", " ", s).strip()


def hash_claim(text: str, urls: List[str]) -> str:
    canon = normalize_text(text).lower() + "\n" + "|".join(sorted(set(urls)))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fetchers and stance
# ---------------------------------------------------------------------------

class Fetcher:
    async def fact_checker_search(self, claim: Claim) -> List[Evidence]:
        return []

    async def general_search(self, claim: Claim) -> List[Evidence]:
        return []

    async def reverse_image(self, claim: Claim) -> List[Evidence]:
        return []


class OpenAIWebFetcher(Fetcher):
    async def fact_checker_search(self, claim: Claim) -> List[Evidence]:
        return await self.general_search(claim)

    async def general_search(self, claim: Claim) -> List[Evidence]:
        logger.debug("Web fetcher: starting search for claim '%s'", claim.text_norm)
        if not getattr(config, "OPENAI_API_KEY", None):
            logger.warning("Web fetcher: OPENAI_API_KEY missing, skipping web search")
            return []
        client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        extra: Dict[str, object] = {}
        if getattr(config, "OPENAI_SEARCH_CONTEXT_SIZE", None):
            extra["search_context_size"] = config.OPENAI_SEARCH_CONTEXT_SIZE
        if getattr(config, "OPENAI_SEARCH_USER_LOCATION", None):
            try:
                extra["user_location"] = json.loads(config.OPENAI_SEARCH_USER_LOCATION)
            except Exception:
                extra["user_location"] = {"country": config.OPENAI_SEARCH_USER_LOCATION}
        items: List[Dict[str, str]] = []
        try:
            # Ask the model to search and return ranked JSON items with stance when possible
            resp = await client.responses.create(
                model=getattr(config, "OPENAI_DEEP_RESEARCH_MODEL_ID", "gpt-4o-mini"),
                tools=[{"type": "web_search"}],
                tool_choice="auto",
                instructions=(
                    "Search the web for the claim and return ONLY JSON as {\"items\": ["
                    "{\"url\": string, \"title\": string}...]]}"
                ),
                input=claim.text_norm,
                **extra,
            )
            text = (getattr(resp, "output_text", "") or "").strip()
            if text.startswith("{"):
                try:
                    data = json.loads(text)
                    items = data.get("items", []) or []
                except Exception:
                    items = []
            if not items and text:
                url_candidates = re.findall(r"https?://[^\s)]+", text)
                for u in url_candidates:
                    items.append({"url": u, "title": ""})
        except Exception as e:
            logger.error("Web fetcher: search failed: %s", e, exc_info=True)
            return []
        evidences: List[Evidence] = []
        # Rank and dedupe by domain
        seen = set()
        for item in items:
            url = item.get("url")
            if not url:
                continue
            domain = (urlparse(url).netloc or url).lower()
            # Skip blocked domains (e.g., t.me)
            base_domain = domain.split(':')[0]
            if base_domain in getattr(config, "FACTCHECK_DOMAIN_BLOCKLIST", set()):
                continue
            if base_domain in seen:
                continue
            seen.add(base_domain)
            title = item.get("title", "")
            evidences.append(
                Evidence(
                    url=url,
                    domain=base_domain,
                    stance="support",
                    note=title or base_domain,
                    published_at=None,
                    snapshot_url=None,
                    tier=None,
                    score=1.0,
                )
            )
            if len(evidences) >= 8:
                break
        return evidences


class StanceModel:
    async def classify(self, claim: Claim, evidences: List[Evidence]) -> List[Evidence]:
        return evidences


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FactChecker:
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
        lang = None
        if detect:
            try:
                lang = detect(text_norm)
            except LangDetectException:
                lang = None
        return Claim(text_norm=text_norm, text_orig=text, lang=lang, urls=urls, hash=hash_claim(text_norm, urls))

    async def extract_quote(self, text: str) -> Optional[Quote]:
        m = re.search(r"[\"«](.+?)[\"»](?:\s*[\u2014\-]\s*([^\n]+))?", text, re.S)
        if not m:
            return None
        quote = normalize_text(m.group(1))
        if " " not in quote:
            return None
        author = title = None
        if m.group(2):
            parts = [p.strip() for p in re.split(r",|\u2014|-", m.group(2), maxsplit=1) if p.strip()]
            if parts:
                author = parts[0]
                if len(parts) > 1:
                    title = parts[1]
        lang = None
        if detect:
            try:
                lang = detect(quote)
            except LangDetectException:
                lang = None
        return Quote(quote=quote, author=author, title=title, lang=lang, hash=hashlib.sha256(quote.lower().encode("utf-8")).hexdigest())

    async def _llm_verdict(self, claim: Claim, debug: List[str], sources: Optional[List[Evidence]] = None) -> Verdict:
        if not self.llm_services:
            debug.append("LLM service unavailable")
            return Verdict(label="unverified", confidence=0.0, summary="LLM service unavailable.", sources=[])

        # Heuristic: consider confirmation if we have enough independent domains
        sources = sources or []
        # Exclude any sources that are from blocked domains
        blocked = getattr(config, "FACTCHECK_DOMAIN_BLOCKLIST", set())
        filtered_sources = [e for e in sources if (e.domain or "").lower() not in blocked]
        distinct_domains = {e.domain for e in filtered_sources}
        confirmed_by_web = len(distinct_domains) >= getattr(config, "FACTCHECK_CONFIRMATION_THRESHOLD", 3)

        # Map to simple labels without emitting JSON in user output
        if confirmed_by_web:
            return Verdict(
                label="true",
                confidence=0.9,
                summary="",
                sources=filtered_sources,
            )

        # Otherwise, ask the model to synthesize a short textual explanation (no JSON) with the web tool enabled
        ctx_lines = [f"- {e.domain}: {e.url}" for e in sources[:5]]
        ctx_block = ("\n\nContext sources (for your reference):\n" + "\n".join(ctx_lines)) if ctx_lines else ""
        system_prompt = (
            "You are a cautious fact-checking assistant. Decide if the claim is likely false or unverified based on the web search results. "
            "Return only a concise explanation in the claim's language. Do not include JSON, labels, or markdown code fences."
        )
        user_prompt = f"Claim: {claim.text_orig}{ctx_block}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        debug.extend([f"System prompt: {system_prompt}", f"User prompt: {user_prompt}"])

        try:
            completion = await self.llm_services.call_openai_deep_research(messages, max_output_tokens=300)
        except Exception as e:
            logger.error("LLM verdict: call failed: %s", e, exc_info=True)
            completion = None

        summary = (completion or "").strip()
        # If the model still emitted JSON fences, strip them defensively
        if summary.startswith("```"):
            summary = summary.strip("` ")
            # Remove a leading json token if present
            if summary.lower().startswith("json"):
                summary = summary[4:].lstrip()
        return Verdict(label="false", confidence=0.7, summary=summary, sources=sources)

    async def _llm_quote_verdict(self, quote: Quote, debug: List[str]) -> Verdict:
        if not self.llm_services:
            debug.append("LLM service unavailable for quote check")
            return Verdict(label="unverified", confidence=0.0, summary="LLM service unavailable.", sources=[])
        system_prompt = (
            "You verify literary quotations. Determine if the provided quote is accurate and correctly attributed. "
            "Respond with JSON: {label: accurate|misquote|misattrib|unverified, confidence: 0-1, summary}."
        )
        user_prompt = f"Quote: \"{quote.quote}\"\nAuthor: {quote.author or 'unknown'}\nTitle: {quote.title or 'unknown'}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        debug.extend([f"System prompt: {system_prompt}", f"User prompt: {user_prompt}"])
        try:
            completion = await self.llm_services.call_openai_llm(messages, temperature=0.0, max_tokens=300)
            debug.append(f"Raw response: {completion}")
        except Exception as e:
            logger.error("LLM quote-check failed: %s", e, exc_info=True)
            completion = None
        label = "unverified"; confidence = 0.0; summary = "No analysis available."
        if completion:
            try:
                data = json.loads(completion)
                label = (data.get("label") or label).strip().lower()
                confidence = float(data.get("confidence", 0.0))
                summary = data.get("summary", summary)
            except Exception:
                summary = completion.strip()
        return Verdict(label=label, confidence=confidence, summary=summary, sources=[])

    async def research(self, claim: Claim | Quote | None, track: str = "news") -> Verdict:
        if not claim:
            return Verdict(label="unverified", confidence=0.0, summary="No claim to check.", sources=[])
        debug: List[str] = [f"track={track}"]
        evidence: List[Evidence] = []
        if isinstance(claim, Claim) and self.fetcher:
            try:
                web_hits = await self.fetcher.fact_checker_search(claim)
                evidence.extend(web_hits)
                debug.append(f"web fetcher returned {len(web_hits)} items")
                print(f"WEB-FETCH primary lang={claim.lang or 'unknown'} hits={len(web_hits)}")
                if web_hits:
                    top_domains = ", ".join({e.domain for e in web_hits[:3]})
                    print(f"WEB-DOMAINS {top_domains}")
            except Exception as e:
                logger.error("Research: web fetcher error: %s", e, exc_info=True)
                debug.append(f"web fetcher error: {e}")
        # Translation fallback if not enough independent confirmations and non-English
        if (
            isinstance(claim, Claim)
            and self.fetcher
            and claim.lang and claim.lang != "en"
            and len({e.domain for e in evidence}) < getattr(config, "FACTCHECK_CONFIRMATION_THRESHOLD", 2)
            and self.llm_services
        ):
            try:
                messages = [
                    {"role": "system", "content": "Translate the following text to English."},
                    {"role": "user", "content": claim.text_norm},
                ]
                translated = await self.llm_services.call_openai_llm(
                    messages,
                    model_id=self.llm_services.openai_translation_model_id,
                    temperature=0.0,
                    max_tokens=1000,
                )
                if translated:
                    t_claim = Claim(
                        text_norm=translated.strip(),
                        text_orig=claim.text_orig,
                        lang="en",
                        urls=claim.urls,
                        hash=claim.hash,
                    )
                    web_hits2 = await self.fetcher.fact_checker_search(t_claim)
                    # Merge any new domains
                    known = {e.domain for e in evidence}
                    for ev in web_hits2:
                        if ev.domain not in known:
                            evidence.append(ev)
                            known.add(ev.domain)
                    debug.append(f"web fetcher returned {len(web_hits2)} items after translation")
                    print(f"WEB-FETCH translated hits={len(web_hits2)}")
            except Exception as e:
                logger.error("Research: translation/web fetch fallback failed: %s", e, exc_info=True)
                debug.append(f"translation/web fetch fallback error: {e}")

        if isinstance(claim, Quote) and track == "book":
            verdict = await self._llm_quote_verdict(claim, debug)
        else:
            verdict = await self._llm_verdict(claim, debug, sources=evidence)  # type: ignore[arg-type]
        verdict.sources = evidence
        verdict.debug = debug
        return verdict


# ---------------------------------------------------------------------------
# Telegram glue
# ---------------------------------------------------------------------------

class FactCheckBot:
    def __init__(
        self,
        app: Application,
        fc: FactChecker,
        satire_detector: Optional[SatireDetector] = None,
        news_gate: Optional[NewsGate] = None,
        quote_gate: Optional[QuoteGate] = None,
        cfg_reader: Callable[[int], Dict[str, object]] | None = None,
        db_manager: Optional[DatabaseManager] = None,
        language_service=None,
    ) -> None:
        self.app = app
        self.fc = fc
        self.satire = satire_detector or SatireDetector(cfg_reader or (lambda _cid: {}))
        self.news_gate = news_gate or NewsGate()
        self.quote_gate = quote_gate or QuoteGate()
        self.cfg_reader = cfg_reader or (lambda _chat_id: {})
        self.db_manager = db_manager
        self.language_service = language_service

    def register(self) -> None:
        self.app.add_handler(CommandHandler("factcheck", self.cmd_factcheck))
        self.app.add_handler(MessageHandler(filters.FORWARDED & TEXT_OR_CAPTION, self.on_forward), group=-1)
        # Safety net for older PTB versions where Caption filter may not fire for media
        try:
            document_filter = filters.DOCUMENT  # PTB < v22
        except AttributeError:
            document_filter = filters.Document.ALL  # PTB v22+
        self.app.add_handler(
            MessageHandler(filters.FORWARDED & (filters.PHOTO | filters.VIDEO | document_filter), self.on_forward),
            group=-1,
        )
        self.app.add_handler(CallbackQueryHandler(self.on_factconfig_cb, pattern=r"^FC:"))
        self.app.add_handler(CommandHandler("factconfig", self.cmd_factconfig))

    async def cmd_factcheck(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """/factcheck command handler. Checks replied message or arguments."""
        if update.effective_message.reply_to_message:
            text = get_text(update.effective_message.reply_to_message) or ""
            target = update.effective_message.reply_to_message
        else:
            text = " ".join(ctx.args)
            target = None
        await self._run_check(update, ctx, text, message=target, track="news")

    async def _run_check(
        self,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        text: str,
        message: Optional[Message] = None,
        track: str = "news",
    ) -> None:
        """Run a fact check and react to the target message."""
        logger.debug("Run check: track=%s text=%r", track, text)
        claim_or_quote = await (self.fc.extract_quote(text) if track == "book" else self.fc.extract_claim(text))
        if not claim_or_quote:
            logger.debug("Run check: claim extraction failed")
            return

        verdict = await self.fc.research(claim_or_quote, track=track)
        # Terminal verdict preview
        try:
            preview = (verdict.summary or "").replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            print(f"VERDICT track={track} label={verdict.label} conf={verdict.confidence:.2f} | {preview}")
        except Exception:
            pass

        label = (verdict.label or "").strip().lower()
        target_msg = message or update.effective_message

        # Persist
        if self.db_manager and update.effective_chat and target_msg:
            try:
                await self.db_manager.log_fact_check(
                    update.effective_chat.id,
                    target_msg.message_id,
                    getattr(claim_or_quote, "text_orig", getattr(claim_or_quote, "quote", "")),
                    label,
                    verdict.confidence,
                    track,
                    "\n".join(verdict.debug) if verdict.debug else None,
                )
            except Exception as e:
                logger.error("Failed to log fact check: %s", e, exc_info=True)

        # React and reply
        try:
            negative_labels = {"false", "misleading_media"}
            positive_labels = {"true", "mostly_true"}
            if label in positive_labels:
                await target_msg.set_reaction("👍")
                # Do not send a message body for positive confirmations
            elif label in negative_labels:
                await target_msg.set_reaction("👎")
                if verdict.summary:
                    try:
                        await target_msg.reply_text(verdict.summary, disable_web_page_preview=True)
                    except BadRequest as br:
                        if "replied not found" in str(br).lower() or "message to be replied not found" in str(br).lower():
                            await update.effective_chat.send_message(verdict.summary)
                        else:
                            raise
            else:
                # needs_context, unverified, opinion -> send a short text if available
                if verdict.summary:
                    try:
                        await target_msg.reply_text(verdict.summary, disable_web_page_preview=True)
                    except BadRequest as br:
                        if "replied not found" in str(br).lower() or "message to be replied not found" in str(br).lower():
                            await update.effective_chat.send_message(verdict.summary)
                        else:
                            raise
        except Exception:
            if label not in {"true", "mostly_true"} and verdict.summary:
                try:
                    await target_msg.reply_text(verdict.summary, disable_web_page_preview=True)
                except Exception:
                    await update.effective_chat.send_message(verdict.summary)

    async def run_direct_check(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str):
        await self._run_check(update, ctx, text, track="news")

    # -------------------------- Handlers ---------------------------------
    async def on_forward(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        # Forward source extraction (legacy + new API)
        forward_from_chat = getattr(message, "forward_from_chat", None)
        forward_origin = getattr(message, "forward_origin", None)
        src_username = None
        src_title = None
        if forward_from_chat:
            src_username = getattr(forward_from_chat, "username", None)
            src_title = getattr(forward_from_chat, "title", None)
        elif forward_origin:
            try:
                origin_chat = getattr(forward_origin, "chat", None) or getattr(forward_origin, "sender_chat", None)
                src_username = getattr(origin_chat, "username", None)
                src_title = getattr(origin_chat, "title", None)
            except Exception:
                pass

        text_for_logging = get_text(message) or ""
        # Terminal mirror of the forward
        print(
            f"FORWARDED chat={getattr(chat,'id',None)} user={getattr(user,'username',None) or getattr(user,'id',None)} "
            f"msg={getattr(message,'message_id',None)} from={src_username or src_title or 'unknown'} "
            f"auto={getattr(message,'is_automatic_forward', None)} | {text_for_logging[:140].replace('\n',' ')}"
        )

        # Only process real forwards
        if not (is_forwarded_message(message) or forward_from_chat or forward_origin):
            return

        # Extract text and try OCR for media; always merge OCR if present
        content_text = text_for_logging
        has_media = bool(getattr(message, 'photo', None) or getattr(message, 'video', None) or getattr(message, 'document', None))
        if has_media:
            ocr_text = await self._ocr_extract(message)
            if ocr_text:
                print(f"OCR merged: +{len(ocr_text)} chars")
                if content_text:
                    # Avoid duplicate long merges
                    if ocr_text not in content_text:
                        content_text = (content_text + "\n" + ocr_text).strip()
                else:
                    content_text = ocr_text

        # Check against known news channels by username
        normalized_username = src_username.lstrip("@").lower() if src_username else None
        is_known_source = False
        if normalized_username and self.db_manager:
            try:
                raw_sources = await self.db_manager.get_news_channel_usernames()
                known = {n.lstrip('@').lower() for n in raw_sources}
                is_known_source = normalized_username in known
            except Exception:
                logger.exception("Failed to load news channel list")
        print(f"SOURCE known={is_known_source} channel={normalized_username}")

        # If known source -> run as news immediately
        if is_known_source:
            await self._run_check(update, ctx, content_text, track="news")
            # Log forwarded message as well
            await self._log_forward_to_db(chat, user, message, text_for_logging)
            return

        # Otherwise run gates
        p_news = await self.news_gate.predict(content_text)
        p_book = await self.quote_gate.predict(content_text)
        _, reasons = self.news_gate.debug_predict(content_text)
        print(f"GATE p_news={p_news:.2f} p_book={p_book:.2f} reasons={'|'.join(reasons) if reasons else 'none'}")

        if p_book >= 0.70:
            print(f"DECISION track=book action=trigger threshold=0.70 score={p_book:.2f}")
            await self._run_check(update, ctx, content_text, track="book")
            await self._log_forward_to_db(chat, user, message, text_for_logging)
            return
        if p_news >= 0.70:
            print(f"DECISION track=news action=trigger threshold=0.70 score={p_news:.2f}")
            await self._run_check(update, ctx, content_text, track="news")
            await self._log_forward_to_db(chat, user, message, text_for_logging)
            return
        if p_book >= 0.55:
            print(f"DECISION track=book action=hint threshold=0.55 score={p_book:.2f}")
            hint = self.language_service.get_response_string("hint_check_quote", "Check quote?") if self.language_service else "Check quote?"
            await self._show_author_only_hint(update, ctx, hint, "book")
            await self._log_forward_to_db(chat, user, message, text_for_logging)
            return
        if p_news >= 0.55:
            print(f"DECISION track=news action=hint threshold=0.55 score={p_news:.2f}")
            hint = self.language_service.get_response_string("hint_check_news", "Check as news?") if self.language_service else "Check as news?"
            await self._show_author_only_hint(update, ctx, hint, "news")
            await self._log_forward_to_db(chat, user, message, text_for_logging)
            return

        # Below thresholds -> ignore, but still log
        print(f"DECISION track=none action=ignore score_news={p_news:.2f} score_book={p_book:.2f} reasons={'|'.join(reasons) if reasons else 'none'}")
        await self._log_forward_to_db(chat, user, message, text_for_logging)

    async def _ocr_extract(self, message: Message) -> str:
        """Extract text from media using OpenAI vision if available. Falls back to empty string."""
        try:
            # Choose media
            if getattr(message, "photo", None):
                ps = message.photo[-1]
                file = await ps.get_file()
            elif getattr(message, "document", None) and str(message.document.mime_type or "").startswith("image/"):
                file = await message.document.get_file()
            elif getattr(message, "video", None) and getattr(message.video, "thumbnail", None):
                file = await message.video.thumbnail.get_file()
            else:
                return ""
            buf = BytesIO()
            await file.download_to_memory(out=buf)
            img_bytes = buf.getvalue()
            if not img_bytes or not self.fc.llm_services:
                return ""
            text = await self.fc.llm_services.extract_text_from_image_bytes(img_bytes)
            return text.strip() if text else ""
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return ""

    async def _log_forward_to_db(self, chat, user, message, text_for_logging: str) -> None:
        """Persist forwarded message in DB if configured."""
        if chat and user and text_for_logging and self.db_manager:
            try:
                await self.db_manager.log_chat_message_and_upsert_user(
                    chat_id=chat.id,
                    user_id=user.id,
                    username=getattr(user, "username", None),
                    first_name=getattr(user, "first_name", None),
                    last_name=getattr(user, "last_name", None),
                    message_id=message.message_id,
                    message_text=text_for_logging,
                    preferred_language=getattr(self.language_service, "current_lang", None),
                )
                print(f"LOGGED-FWD chat={chat.id} user={user.username or user.id} msg={message.message_id}")
            except Exception as exc:
                logger.error("Failed to log forwarded message: %s", exc)

    async def _show_author_only_hint(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, hint: str, track: str) -> None:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(hint, callback_data=f"FC:GATE:CHECK:{track}")]])
        try:
            await ctx.bot.send_message(update.effective_user.id, hint, reply_markup=kb)
        except Exception:
            pass

    def _tr(self, key: str, default: str) -> str:
        if self.language_service and hasattr(self.language_service, "get_response_string"):
            try:
                return self.language_service.get_response_string(key, default)
            except Exception:
                return default
        return default

    async def cmd_factconfig(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(self._tr("factconfig_tab_preset", "Preset"), callback_data="FC:TAB:Preset"),
                    InlineKeyboardButton(self._tr("button_show_sources", "Sources"), callback_data="FC:TAB:Sources"),
                ],
                [
                    InlineKeyboardButton(self._tr("factconfig_tab_policy", "Policy"), callback_data="FC:TAB:Policy"),
                    InlineKeyboardButton(self._tr("factconfig_tab_limits", "Limits"), callback_data="FC:TAB:Limits"),
                ],
                [
                    InlineKeyboardButton(self._tr("factconfig_tab_auto", "Auto"), callback_data="FC:TAB:Auto"),
                    InlineKeyboardButton(self._tr("factconfig_tab_danger", "Danger"), callback_data="FC:TAB:Danger"),
                ],
                [
                    InlineKeyboardButton(self._tr("factconfig_export_btn", "Export"), callback_data="FC:EXPORT"),
                    InlineKeyboardButton(self._tr("factconfig_apply_btn", "Apply"), callback_data="FC:APPLY"),
                ],
            ]
        )
        await update.effective_message.reply_text(self._tr("factconfig_title", "Fact check config:"), reply_markup=kb)

    async def on_factconfig_cb(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        data = q.data
        await q.answer()
        if data == "FC:FORCE":
            if q.message and q.message.reply_to_message:
                orig = q.message.reply_to_message
                text = get_text(orig) or ""
                await q.edit_message_text(self._tr("factconfig_checking", "Checking…"))
                await self._run_check(update, ctx, text, message=orig)
            else:
                await q.edit_message_text(self._tr("factconfig_nothing_to_check", "Nothing to check."))
            return
        if data.startswith("FC:GATE:CHECK"):
            parts = data.split(":")
            track = parts[3] if len(parts) > 3 else "news"
            text = (q.message.text or "").replace("\n\n" + self._tr("hint_check_quote", "Check quote?"), "").replace("\n\n" + self._tr("hint_check_news", "Check as news?"), "").strip()
            await q.edit_message_text(self._tr("factconfig_checking", "Checking…"))
            await self._run_check(update, ctx, text, track=track)
            return
        if data == "FC:EXPORT":
            await q.edit_message_text(self._tr("factconfig_export_ok", "Configuration exported."))
            return
        if data == "FC:APPLY":
            await q.edit_message_text(self._tr("factconfig_apply_ok", "Configuration applied."))
            return
        await q.edit_message_text(self._tr("factconfig_update_ok", "Config updated."))

    async def _log_satire(self, update: Update, dec: SatireDecision) -> None:
        try:
            debug_data = json.dumps(dec.rationale)
        except Exception:  # pragma: no cover
            debug_data = "{}"
        update_str = f"chat={update.effective_chat.id} msg={update.effective_message.message_id}"
        print(f"Satire decision {dec.decision} for {update_str}: {debug_data}")
