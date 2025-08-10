# enkibot/modules/community_moderation.py
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
"""Community moderation helpers.

This module implements a lightweight, in-memory community moderation
system inspired by more advanced crowd‑moderation examples.  It supports
reporting with a reason picker, weighted voting with basic trust scores
and automatic actions once dynamic thresholds are met.  The state is not
persisted between restarts but provides a foundation that can later be
extended to a database-backed implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import ContextTypes


# ---------------------------------------------------------------------------
# Constants and simple helpers
# ---------------------------------------------------------------------------

CONSENSUS_WINDOW_MIN = 15
TRUST_MIN, TRUST_MAX = 0.2, 1.5
VOTE_LIMIT_PER_DAY = 10
MIN_CHAT_TENURE_HOURS = 24

THRESH_HIDE = 2.0
THRESH_TEMPBAN = 5.0
THRESH_PERMABAN = 7.0

TEMPBAN_HOURS = 24

REASON_CHOICES = [
    ("spam", "Spam/Ads"),
    ("scam", "Scam/Phishing"),
    ("nsfw", "NSFW"),
    ("hate", "Hate/Harassment"),
    ("offtopic", "Off-topic"),
    ("other", "Other"),
]


def now() -> datetime:
    return datetime.utcnow()


@dataclass
class Case:
    case_id: int
    chat_id: int
    target_user_id: int
    first_msg_id: int
    opened_ts: datetime
    reason: str
    votes: Dict[int, float] = field(default_factory=dict)
    status: str = "open"


class CommunityModerationService:
    """Minimal community moderation manager."""

    def __init__(self, admin_chat_id: Optional[int] = None) -> None:
        self.admin_chat_id = admin_chat_id
        self.cases: Dict[int, Case] = {}
        self.case_index: Dict[Tuple[int, int, int], int] = {}
        self.next_case_id = 1
        self.trust: Dict[Tuple[int, int], float] = {}
        self.mem_state: Dict[Tuple[int, int], Dict[str, any]] = {}

    # ------------------------------------------------------------------
    # Case and trust helpers
    # ------------------------------------------------------------------
    def _get_or_open_case(self, chat_id: int, target_user_id: int, msg_id: int, reason: str) -> Case:
        key = (chat_id, target_user_id, msg_id)
        cid = self.case_index.get(key)
        if cid:
            return self.cases[cid]
        case = Case(self.next_case_id, chat_id, target_user_id, msg_id, now(), reason)
        self.cases[self.next_case_id] = case
        self.case_index[key] = self.next_case_id
        self.next_case_id += 1
        return case

    def _already_voted(self, case: Case, voter_id: int) -> bool:
        return voter_id in case.votes

    def _add_vote(self, case: Case, voter_id: int, base: float = 1.0) -> float:
        trust = self.trust.get((case.chat_id, voter_id), 1.0)
        trust = max(TRUST_MIN, min(TRUST_MAX, trust))
        weight = base * trust
        case.votes[voter_id] = weight
        return weight

    def _effective_score(self, case: Case) -> float:
        window_start = now() - timedelta(minutes=CONSENSUS_WINDOW_MIN)
        return sum(weight for weight in case.votes.values() if window_start <= case.opened_ts)

    # ------------------------------------------------------------------
    # Eligibility tracking
    # ------------------------------------------------------------------
    def _eligible_to_vote(self, update: Update) -> Tuple[bool, str]:
        chat = update.effective_chat
        user = update.effective_user
        key = (chat.id, user.id)
        st = self.mem_state.get(key, {})

        joined_ts = st.get("joined_ts")
        if not joined_ts:
            joined_ts = now()
            st["joined_ts"] = joined_ts
        if now() - joined_ts < timedelta(hours=MIN_CHAT_TENURE_HOURS):
            self.mem_state[key] = st
            return False, "You need more time in this chat before voting."

        votes = [t for t in st.get("votes", []) if now() - t < timedelta(days=1)]
        if len(votes) >= VOTE_LIMIT_PER_DAY:
            st["votes"] = votes
            self.mem_state[key] = st
            return False, "Daily vote limit reached."
        st["votes"] = votes
        self.mem_state[key] = st
        return True, ""

    def _record_vote_usage(self, update: Update) -> None:
        chat = update.effective_chat
        user = update.effective_user
        key = (chat.id, user.id)
        st = self.mem_state.get(key, {})
        votes = st.get("votes", [])
        votes.append(now())
        st["votes"] = votes
        self.mem_state[key] = st

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    async def _hide_message(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_id: int) -> None:
        try:
            await context.bot.delete_message(chat_id, msg_id)
        except Exception:
            pass

    async def _temp_ban(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, hours: int = TEMPBAN_HOURS) -> None:
        try:
            until = int((now() + timedelta(hours=hours)).timestamp())
            await context.bot.ban_chat_member(chat_id, user_id, until_date=until)
        except Exception:
            pass

    async def _permaban(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> None:
        try:
            await context.bot.ban_chat_member(chat_id, user_id)
        except Exception:
            pass

    async def _apply_decision(self, context: ContextTypes.DEFAULT_TYPE, case: Case, score: float, msg_id: Optional[int]) -> Optional[str]:
        action = None
        if score >= THRESH_PERMABAN:
            action = "permaban"
        elif score >= THRESH_TEMPBAN:
            action = "tempban"
        elif score >= THRESH_HIDE:
            action = "hide"

        if not action:
            return None

        if action == "hide" and msg_id:
            await self._hide_message(context, case.chat_id, msg_id)
        elif action == "tempban":
            await self._temp_ban(context, case.chat_id, case.target_user_id)
        elif action == "permaban":
            await self._permaban(context, case.chat_id, case.target_user_id)

        if self.admin_chat_id:
            try:
                await context.bot.send_message(
                    self.admin_chat_id,
                    f"Case #{case.case_id} action: {action} (score {score:.2f})",
                )
            except Exception:
                pass
        return action

    # ------------------------------------------------------------------
    # Public command handlers
    # ------------------------------------------------------------------
    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to a message with /report to report it.")
            return
        buttons = [
            [InlineKeyboardButton(lbl, callback_data=f"REPORT:{code}:{update.message.reply_to_message.message_id}")]
            for code, lbl in REASON_CHOICES
        ]
        kb = InlineKeyboardMarkup(buttons)
        await update.effective_message.reply_text("Choose a reason to report:", reply_markup=kb)

    async def on_report_reason(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        q = update.callback_query
        await q.answer()
        try:
            _, reason, msg_id_str = q.data.split(":", 2)
            msg_id = int(msg_id_str)
        except Exception:
            return

        ok, why = self._eligible_to_vote(update)
        if not ok:
            await q.edit_message_text(why)
            return

        chat = update.effective_chat
        user = update.effective_user
        try:
            orig = await context.bot.forward_message(chat.id, chat.id, msg_id)
            target_id = orig.forward_from.id if orig.forward_from else None
        except Exception:
            target_id = None
        if not target_id:
            await q.edit_message_text("Could not identify target user.")
            return

        case = self._get_or_open_case(chat.id, target_id, msg_id, reason)
        if self._already_voted(case, user.id):
            await q.edit_message_text("You already reported this case.")
            return
        self._add_vote(case, user.id, base=1.0)
        self._record_vote_usage(update)
        score = self._effective_score(case)
        await q.edit_message_text("Thanks, the moderators have been notified.")
        await self._apply_decision(context, case, score, msg_id)
        if self.admin_chat_id:
            try:
                await context.bot.send_message(
                    self.admin_chat_id,
                    f"Report: case {case.case_id} voter {user.id} reason {reason} score {score:.2f}",
                )
            except Exception:
                pass

    async def cmd_vote(self, update: Update, context: ContextTypes.DEFAULT_TYPE, reason: str = "spam") -> None:
        if not update.message or not update.message.reply_to_message:
            await update.effective_message.reply_text("Reply to the offending message with /spam.")
            return
        voter = update.effective_user
        target = update.message.reply_to_message.from_user
        if voter.id == target.id:
            await update.effective_message.reply_text("You cannot vote on yourself.")
            return

        ok, why = self._eligible_to_vote(update)
        if not ok:
            await update.effective_message.reply_text(why)
            return

        case = self._get_or_open_case(update.effective_chat.id, target.id, update.message.reply_to_message.message_id, reason)
        if self._already_voted(case, voter.id):
            await update.effective_message.reply_text("You already voted on this case.")
            return
        w = self._add_vote(case, voter.id, base=1.0)
        self._record_vote_usage(update)
        score = self._effective_score(case)
        await update.effective_message.reply_text("Vote recorded. Mods notified.")
        await self._apply_decision(context, case, score, update.message.reply_to_message.message_id)
        if self.admin_chat_id:
            try:
                await context.bot.send_message(
                    self.admin_chat_id,
                    f"Vote: case {case.case_id} voter {voter.id} +{w:.2f} score {score:.2f}",
                )
            except Exception:
                pass
