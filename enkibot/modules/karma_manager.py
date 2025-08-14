# enkibot/modules/karma_manager.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

from enkibot import config
from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class KarmaWeights:
    plus_command: float = 1.0
    plus_reaction: float = 0.5
    minus_command: float = -1.0
    minus_reaction: float = -0.5


class KarmaManager:
    """DB-backed karma manager implementing weighted votes, cooldowns, and decay.

    This class provides the minimal API expected by TelegramHandlerService:
    - handle_text_vote(giver_id, receiver_id, chat_id, message_text)
    - record_reaction_event(chat_id, msg_id, target_user_id, rater_user_id, emoji)
    - get_user_stats(user_id)
    - get_top_users(chat_id)
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.weights = KarmaWeights(
            plus_command=float(getattr(config, "KARMA_PLUS_COMMAND", 1.0)),
            plus_reaction=float(getattr(config, "KARMA_PLUS_REACTION", 0.5)),
            minus_command=-abs(float(getattr(config, "KARMA_MINUS_COMMAND", 1.0))),
            minus_reaction=-abs(float(getattr(config, "KARMA_MINUS_REACTION", 0.5))),
        )
        self.cooldown_s = int(getattr(config, "KARMA_COOLDOWN_SECONDS", 60))
        self.user_halflife_days = int(getattr(config, "KARMA_DECAY_USER_HALFLIFE_DAYS", 30))
        # In-memory cooldown: (chat_id, rater_id, target_id) -> last_ts
        self._last_tx: Dict[Tuple[int, int, int], float] = {}
        # Emoji mapping (can be made configurable later)
        self._pos_emojis = set(getattr(config, "KARMA_POS_EMOJIS", ["👍", "❤️", "👏", "🔥"]))
        self._neg_emojis = set(getattr(config, "KARMA_NEG_EMOJIS", ["👎", "💩"]))

    # -------------------- Internal helpers --------------------
    def _now(self) -> float:
        return time.time()

    def _lambda_from_halflife(self, days: int) -> float:
        days = max(1, days)
        return math.log(2.0) / float(days)

    def _is_on_cooldown(self, chat_id: int, rater_id: int, target_id: int) -> bool:
        key = (chat_id, rater_id, target_id)
        last = self._last_tx.get(key, 0.0)
        if self._now() - last < self.cooldown_s:
            return True
        self._last_tx[key] = self._now()
        return False

    async def _upsert_rep(self, chat_id: int, user_id: int, delta: float) -> None:
        await self.db.execute_query(
            (
                "MERGE user_rep_current AS t USING (VALUES(?,?,?)) AS s(chat_id,user_id,rep) "
                "ON (t.chat_id=s.chat_id AND t.user_id=s.user_id) "
                "WHEN MATCHED THEN UPDATE SET rep = t.rep + s.rep, last_seen = SYSUTCDATETIME() "
                "WHEN NOT MATCHED THEN INSERT (chat_id,user_id,rep,last_seen) VALUES (s.chat_id,s.user_id,s.rep,SYSUTCDATETIME())"
            ),
            (chat_id, user_id, float(delta)),
            commit=True,
        )

    async def _insert_event(
        self,
        chat_id: int,
        msg_id: Optional[int],
        target_user_id: int,
        rater_user_id: int,
        base: float,
        emoji: Optional[str] = None,
    ) -> None:
        # MVP multipliers (kept at 1.0 for now)
        rater_trust = 1.0
        diversity = 1.0
        anti_collusion = 1.0
        novelty = 1.0
        content_factor = 1.0
        weight = base * rater_trust * diversity * anti_collusion * novelty * content_factor
        await self.db.execute_query(
            (
                "INSERT INTO karma_events (chat_id,msg_id,target_user_id,rater_user_id,emoji,base,rater_trust,diversity,anti_collusion,novelty,content_factor,weight) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
            ),
            (
                chat_id,
                msg_id,
                target_user_id,
                rater_user_id,
                emoji,
                base,
                rater_trust,
                diversity,
                anti_collusion,
                novelty,
                content_factor,
                weight,
            ),
            commit=True,
        )
        await self._upsert_rep(chat_id, target_user_id, weight)

    # -------------------- Public API used by TelegramHandlerService --------------------
    async def handle_text_vote(
        self, giver_id: int, receiver_id: int, chat_id: int, message_text: str
    ) -> Optional[Tuple[str, float]]:
        """Parses + / - tokens in a reply and persists the vote.
        Returns (result_code, delta) or None if not a karma token.
        """
        if receiver_id == giver_id:
            return ("self_karma_error", 0.0)
        token = (message_text or "").strip()
        if token not in {"+1", "+", "++", "-1", "-", "--"}:
            return None
        if self._is_on_cooldown(chat_id, giver_id, receiver_id):
            return ("cooldown_error", 0.0)
        base = 0.0
        if token in {"+1", "+", "++"}:
            base = self.weights.plus_command
        elif token in {"-1", "-", "--"}:
            base = self.weights.minus_command
        try:
            await self._insert_event(
                chat_id=chat_id,
                msg_id=None,
                target_user_id=receiver_id,
                rater_user_id=giver_id,
                base=base,
                emoji=None,
            )
            return ("karma_changed_success", base)
        except Exception as e:
            logger.error(f"Karma text vote failed: {e}", exc_info=True)
            return ("error", 0.0)

    async def record_reaction_event(
        self,
        chat_id: int,
        msg_id: Optional[int],
        target_user_id: int,
        rater_user_id: int,
        emoji: str,
    ) -> None:
        if target_user_id == rater_user_id:
            return
        if self._is_on_cooldown(chat_id, rater_user_id, target_user_id):
            return
        if emoji in self._pos_emojis:
            base = self.weights.plus_reaction
        elif emoji in self._neg_emojis:
            base = self.weights.minus_reaction
        else:
            return  # unsupported emoji
        try:
            await self._insert_event(
                chat_id=chat_id,
                msg_id=msg_id,
                target_user_id=target_user_id,
                rater_user_id=rater_user_id,
                base=base,
                emoji=emoji,
            )
        except Exception as e:
            logger.error(f"Failed to record reaction event: {e}", exc_info=True)

    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, float]]:
        """Returns aggregated current rep for the user across all chats."""
        row = await self.db.execute_query(
            "SELECT SUM(rep) AS total FROM user_rep_current WHERE user_id = ?",
            (user_id,),
            fetch_one=True,
        )
        total = float(getattr(row, "total", 0.0)) if row else 0.0
        return {"received": total}

    async def get_top_users(self, chat_id: int, limit: int = 10) -> List[Dict[str, float]]:
        rows = await self.db.execute_query(
            "SELECT TOP (?) user_id, rep, last_seen FROM user_rep_current WHERE chat_id = ? ORDER BY rep DESC",
            (limit * 5, chat_id),
            fetch_all=True,
        )
        if not rows:
            return []
        # Apply exponential decay on-the-fly for display
        from datetime import datetime, timezone
        lam = self._lambda_from_halflife(self.user_halflife_days)
        now = datetime.now(timezone.utc)
        tmp: List[Tuple[int, float]] = []
        for r in rows:
            rep = float(getattr(r, "rep", 0.0) or 0.0)
            last_seen = getattr(r, "last_seen", None)
            if last_seen is None:
                decayed = rep
            else:
                dt_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
                decayed = rep * math.exp(-lam * dt_days)
            uid = int(getattr(r, "user_id", 0))
            tmp.append((uid, decayed))
        tmp.sort(key=lambda x: x[1], reverse=True)
        top = tmp[:limit]
        return [{"user_id": uid, "name": str(uid), "score": round(score, 2)} for uid, score in top]
