# enkibot/modules/spam_detector.py
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
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
"""Advanced spam detection with a Zero-Trust approach."""

from __future__ import annotations

import logging
import re
from time import time
from typing import Any, Callable, Dict, Optional

import tldextract
from telegram import ChatPermissions, Update
from telegram.ext import ContextTypes

from enkibot import config as bot_config
from enkibot.modules.base_module import BaseModule
from enkibot.core.llm_services import LLMServices
from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class SpamDetector(BaseModule):
    """Detects spam or disallowed content and takes moderation actions."""

    def __init__(
        self,
        llm_services: LLMServices,
        db_manager: Optional[DatabaseManager] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__("SpamDetector")
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.enabled = enabled
        self.captcha_callback: Optional[Callable[[Any, int, ContextTypes.DEFAULT_TYPE], Any]] = None
        # Track per-user state across chats: joined timestamp, verification and clean message count
        self.user_states: Dict[tuple[int, int], Dict[str, Any]] = {}
        logger.info("SpamDetector initialized. Enabled=%s", self.enabled)

    # ------------------------------------------------------------------
    # Configuration helpers
    def set_captcha_callback(self, callback: Callable[[Any, int, ContextTypes.DEFAULT_TYPE], Any]) -> None:
        """Allows external components to provide a captcha starter."""
        self.captcha_callback = callback

    def _is_new_or_unverified(self, state: Dict[str, Any]) -> bool:
        cfg = bot_config.ZERO_TRUST_SETTINGS
        if not state:
            return True
        if state.get("verified"):
            return False
        if time() - state.get("joined_ts", 0) <= cfg["watch_new_user_window_sec"]:
            return True
        return state.get("clean_msgs", 0) < cfg["watch_first_messages"]

    def _apply_heuristics(self, text: str) -> float:
        cfg = bot_config.ZERO_TRUST_SETTINGS
        heur = cfg["heuristics"]
        lists = cfg["lists"]
        risk = 0.0
        t = text or ""
        urls = re.findall(r"https?://\S+|t\.me/\S+|\S+\.\S+", t, flags=re.I)
        if urls:
            risk += heur["url_in_first_msg"]
            for u in urls:
                dom = tldextract.extract(u)
                fqdn = ".".join(part for part in [dom.subdomain, dom.domain, dom.suffix] if part)
                if fqdn in lists["domain_blocklist"]:
                    risk += 0.10
        if len(re.findall(r"@\w{3,}", t)) >= 3:
            risk += heur["many_mentions"]
        letters = re.findall(r"[A-Za-z]", t)
        if len(t) > 40 and letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio >= 0.8:
                risk += heur["all_caps_long"]
        if re.search(r"(.)\1{3,}", t) or re.search(r"([!?$€£¥]){4,}", t):
            risk += heur["repeated_chars"]
        low = t.lower()
        if any(k in low for k in lists["keyword_blocklist"]):
            risk += heur["keyword_hits"]
        return min(risk, 1.0)

    def _combined_risk(self, spamish: float, heur_score: float, state: Dict[str, Any]) -> float:
        base = max(spamish, heur_score)
        if self._is_new_or_unverified(state):
            base = min(base + 0.10, 1.0)
        return base

    async def _log_action(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        user,
        text: str,
        risk: float,
        scores: Dict[str, float],
        action: str,
    ) -> None:
        cfg = bot_config.ZERO_TRUST_SETTINGS
        admin_chat_id = cfg["logging"].get("admin_chat_id")
        if not admin_chat_id:
            return
        short = (text[:200] + "…") if text and len(text) > 200 else (text or "")
        try:
            await context.bot.send_message(
                admin_chat_id,
                (
                    f"🚫 Zero-Trust {action}\n"
                    f"• Chat: {chat_id}\n"
                    f"• User: @{getattr(user, 'username', user.id)}\n"
                    f"• Risk: {risk:.2f}\n"
                    f"• Scores: {scores}\n"
                    f"• Excerpt: {short}"
                ),
                disable_web_page_preview=True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    async def inspect_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Checks a message and removes it if deemed risky.

        Returns ``True`` if a message was flagged and handled
        (deleted/banned/muted)."""
        if not self.enabled:
            return False
        if not update.message or not update.message.text:
            return False
        user = update.effective_user
        if not user or user.is_bot:
            return False

        chat_id = update.effective_chat.id
        text = update.message.text
        key = (chat_id, user.id)
        state = self.user_states.get(key)
        if state is None:
            state = {"joined_ts": time(), "verified": False, "clean_msgs": 0}
            self.user_states[key] = state

        heur_score = self._apply_heuristics(text)
        moderation_result = await self.llm_services.moderate_text_openai(text)
        scores_obj = moderation_result.get("category_scores", {}) if moderation_result else {}
        if hasattr(scores_obj, "model_dump"):
            scores = scores_obj.model_dump()
        elif hasattr(scores_obj, "dict"):
            scores = scores_obj.dict()
        elif isinstance(scores_obj, dict):
            scores = scores_obj
        else:
            scores = {}
        spamish = max(scores.values()) if scores else 0.0
        risk = self._combined_risk(spamish, heur_score, state)
        if moderation_result and moderation_result.get("flagged"):
            risk = max(risk, bot_config.ZERO_TRUST_SETTINGS["global_thresholds"]["delete"])

        thresholds = bot_config.ZERO_TRUST_SETTINGS["global_thresholds"]
        action: Optional[str] = None
        if risk >= thresholds["ban"]:
            action = "banned"
        elif risk >= thresholds["mute_then_captcha"]:
            action = "muted_captcha"
        elif risk >= thresholds["delete"]:
            action = "deleted"

        if action:
            message_id = update.message.message_id
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception as e:
                logger.error("Failed to delete message %s in chat %s: %s", message_id, chat_id, e)

            if self.db_manager and self.db_manager.connection_string and moderation_result:
                try:
                    categories_obj = moderation_result.get("categories", {})
                    if hasattr(categories_obj, "model_dump"):
                        categories = categories_obj.model_dump()
                    elif hasattr(categories_obj, "dict"):
                        categories = categories_obj.dict()
                    elif isinstance(categories_obj, dict):
                        categories = categories_obj
                    else:
                        categories = {}
                    flagged = ", ".join([k for k, v in categories.items() if v])
                    await self.db_manager.log_moderation_action(
                        chat_id=chat_id,
                        user_id=user.id,
                        message_id=message_id,
                        categories=flagged or None,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error("Failed to log moderation action: %s", e, exc_info=True)

            if action == "banned":
                try:
                    await context.bot.ban_chat_member(chat_id=chat_id, user_id=user.id)
                except Exception as e:
                    logger.error("Failed to ban user %s in chat %s: %s", user.id, chat_id, e)
            elif action == "muted_captcha":
                try:
                    perms = ChatPermissions(can_send_messages=False)
                    await context.bot.restrict_chat_member(chat_id, user.id, permissions=perms)
                except Exception as e:
                    logger.error("Failed to restrict user %s in chat %s: %s", user.id, chat_id, e)
                if self.captcha_callback:
                    try:
                        await self.captcha_callback(user, chat_id, context)
                    except Exception as e:
                        logger.error("Failed to start captcha for %s: %s", user.id, e)

            await self._log_action(context, chat_id, user, text, risk, scores, action)
            return True

        # Message deemed clean -> update state
        state["clean_msgs"] = state.get("clean_msgs", 0) + 1
        if state["clean_msgs"] >= bot_config.ZERO_TRUST_SETTINGS["watch_first_messages"]:
            state["verified"] = True
        return False
