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

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from ..utils.message_utils import get_text
import httpx
import logging
from ..utils.database import DatabaseManager

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
    """Tiny orchestrator coordinating fetchers and stance model."""

    def __init__(self, fetcher: Fetcher, stance: StanceModel):
        self.fetcher = fetcher
        self.stance = stance

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

    async def research(self, claim: Claim) -> Verdict:
        tasks = [
            asyncio.create_task(self.fetcher.fact_checker_search(claim)),
            asyncio.create_task(self.fetcher.general_search(claim)),
        ]
        fc, web = await asyncio.gather(*tasks)
        evidences = await self.stance.classify(claim, (fc or []) + (web or []))

        score = 0.0
        pos = neg = 0
        for e in evidences:
            if e.stance == "support":
                pos += 1
                score += e.score
            elif e.stance == "refute":
                neg += 1
                score -= e.score
        confidence = min(1.0, max(0.0, abs(score) / max(1, len(evidences))))
        if pos >= 2 and neg == 0 and confidence >= 0.85:
            label = "true"
        elif pos >= 1 and neg == 0 and confidence >= 0.70:
            label = "mostly_true"
        elif neg >= 2 and confidence >= 0.80:
            label = "false"
        elif pos == 0 and neg == 0:
            label = "unverified"
        else:
            label = "needs_context"

        summary = self._make_summary(claim, label, evidences)
        return Verdict(label=label, confidence=confidence, summary=summary, sources=evidences[:6])

    def _make_summary(self, claim: Claim, label: str, evidences: List[Evidence]) -> str:
        lead = {
            "true": "Corroborated by multiple independent sources.",
            "mostly_true": "Gist is correct; minor caveats apply.",
            "false": "Contradicted by reliable sources.",
            "needs_context": "Claim omits key context that changes interpretation.",
            "unverified": "Insufficient credible coverage yet.",
            "misleading_media": "Real media used out of context or edited.",
            "opinion": "Value judgment; not checkable.",
        }.get(label, "Assessment available.")

        if label == "unverified":
            total = len(evidences)
            return (
                f"{lead} "
                "Fact-check and web searches yielded no reliable sources to verify or refute the claim. "
                f"Checked {total} candidate source{'s' if total != 1 else ''}."
            )

        return f"{lead}"


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
        cfg_reader: Callable[[int], Dict[str, object]] | None = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.app = app
        self.fc = fc
        self.satire = satire_detector or SatireDetector(lambda _chat_id: {})
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
            await self._run_check(update, ctx, text)

    async def cmd_factcheck(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_message.reply_to_message:
            text = get_text(update.effective_message.reply_to_message) or ""
        else:
            text = " ".join(ctx.args)
        await self._run_check(update, ctx, text)

    async def _run_check(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        """Run a fact check and react to the original message."""

        claim = await self.fc.extract_claim(text)
        if not claim:
            return

        verdict = await self.fc.research(claim)

        if self.db_manager and update.effective_chat and update.effective_message:
            try:
                await self.db_manager.log_fact_check(
                    update.effective_chat.id,
                    update.effective_message.message_id,
                    claim.text_orig,
                    verdict.label,
                    verdict.confidence,
                )
            except Exception as e:
                logger.error(f"Failed to log fact check: {e}", exc_info=True)

        try:
            if verdict.label in ("true", "mostly_true"):
                await update.effective_message.set_reaction("👍")
            else:
                await update.effective_message.set_reaction("👎")
                await update.effective_message.reply_text(
                    verdict.summary, disable_web_page_preview=True
                )
        except Exception:  # pragma: no cover - reaction support may vary
            if verdict.label not in ("true", "mostly_true"):
                await update.effective_message.reply_text(
                    verdict.summary, disable_web_page_preview=True
                )

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
        await q.answer()
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
