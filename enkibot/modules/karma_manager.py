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
    - get_global_top()
    - diagnose(chat_id)
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

    # -------------------- Diagnostics --------------------
    async def diagnose(self, chat_id: Optional[int] = None) -> str:
        """Run detailed terminal diagnostics for the karma subsystem.

        Prints detailed status to the terminal and returns a short summary string
        suitable for replying in chat.
        """
        lines: List[str] = []
        print("KARMA-DIAG: starting diagnostics...")
        if not getattr(self.db, "connection_string", None):
            msg = "Database connection string is not configured."
            print(f"KARMA-DIAG: {msg}")
            return msg
        conn = self.db.get_db_connection()
        if not conn:
            msg = "Database connection could not be established (see logs)."
            print(f"KARMA-DIAG: {msg}")
            return msg
        try:
            with conn.cursor() as cursor:
                # Check required tables
                req_tables = [
                    "user_rep_current",
                    "karma_events",
                    "message_scores",
                    "user_rep_rollup",
                ]
                missing: List[str] = []
                for t in req_tables:
                    cursor.execute(
                        "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
                        (t,),
                    )
                    if not cursor.fetchone():
                        missing.append(t)
                print(f"KARMA-DIAG: required tables missing: {missing if missing else 'none'}")
                if missing:
                    lines.append("Missing tables: " + ", ".join(missing))
                # Column checks for user_rep_current
                cursor.execute(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'user_rep_current'"
                )
                cols = {r.COLUMN_NAME.lower() for r in cursor.fetchall()} if cursor.description else set()
                needed_cols = {"chat_id", "user_id", "rep", "last_seen"}
                missing_cols = sorted(list(needed_cols - cols)) if cols else list(needed_cols)
                print(f"KARMA-DIAG: user_rep_current columns: {sorted(cols) if cols else 'n/a'}")
                if missing_cols:
                    print(f"KARMA-DIAG: user_rep_current missing columns: {missing_cols}")
                    lines.append("user_rep_current missing columns: " + ", ".join(missing_cols))
                # Row counts
                def _count(q: str, params: tuple = ()) -> int:
                    try:
                        cursor.execute(q, params) if params else cursor.execute(q)
                        row = cursor.fetchone()
                        return int(row[0]) if row else 0
                    except Exception as e:
                        print(f"KARMA-DIAG: count failed for query {q}: {e}")
                        return -1
                total_rep_rows = _count("SELECT COUNT(*) FROM user_rep_current")
                total_events = _count("SELECT COUNT(*) FROM karma_events")
                print(f"KARMA-DIAG: user_rep_current rows={total_rep_rows} | karma_events rows={total_events}")
                if chat_id is not None:
                    per_chat_rows = _count("SELECT COUNT(*) FROM user_rep_current WHERE chat_id = ?", (chat_id,))
                    ev_per_chat = _count("SELECT COUNT(*) FROM karma_events WHERE chat_id = ?", (chat_id,))
                    print(f"KARMA-DIAG: chat {chat_id}: rep_rows={per_chat_rows} | events={ev_per_chat}")
                    lines.append(f"chat {chat_id}: rep_rows={per_chat_rows}, events={ev_per_chat}")
                # Sample rows
                try:
                    cursor.execute(
                        "SELECT TOP 5 chat_id, user_id, rep, last_seen FROM user_rep_current ORDER BY rep DESC"
                    )
                    sample = cursor.fetchall()
                    print("KARMA-DIAG: sample user_rep_current top5:")
                    for r in sample or []:
                        print(f"  chat={getattr(r,'chat_id',None)} user={getattr(r,'user_id',None)} rep={getattr(r,'rep',None)} last={getattr(r,'last_seen',None)}")
                except Exception as e:
                    print(f"KARMA-DIAG: sample read failed: {e}")
                # Final summary
                if total_events == 0 and total_rep_rows == 0:
                    lines.append("No karma data recorded yet.")
                elif total_events > 0 and total_rep_rows == 0:
                    lines.append("Events exist but user_rep_current is empty. Rep rollup may be missing.")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        summary = "; ".join(lines) if lines else "Karma diagnostics: OK."
        print(f"KARMA-DIAG: summary -> {summary}")
        return summary

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
        # Diagnostics
        print(f"KARMA-UPsert chat={chat_id} user={user_id} delta={delta}")

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
            print(f"KARMA-REACTION skip=self chat={chat_id} msg={msg_id} user={rater_user_id} emoji={emoji}")
            return
        if self._is_on_cooldown(chat_id, rater_user_id, target_user_id):
            print(f"KARMA-REACTION skip=cooldown chat={chat_id} msg={msg_id} target={target_user_id} rater={rater_user_id} emoji={emoji}")
            return
        if emoji in self._pos_emojis:
            base = self.weights.plus_reaction
            reason = "pos"
        elif emoji in self._neg_emojis:
            base = self.weights.minus_reaction
            reason = "neg"
        else:
            print(f"KARMA-REACTION skip=unsupported-emoji chat={chat_id} msg={msg_id} target={target_user_id} rater={rater_user_id} emoji={emoji}")
            return  # unsupported emoji
        try:
            print(f"KARMA-REACTION insert chat={chat_id} msg={msg_id} target={target_user_id} rater={rater_user_id} emoji={emoji} base={base} reason={reason}")
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
            print(f"KARMA-REACTION error={e}")

    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, float]]:
        """Returns aggregated current rep for the user across all chats (for simple notifications)."""
        row = await self.db.execute_query(
            "SELECT SUM(rep) AS total FROM user_rep_current WHERE user_id = ?",
            (user_id,),
            fetch_one=True,
        )
        total = float(getattr(row, "total", 0.0) if row else 0.0)
        return {"received": total}

    async def get_user_stats_in_chat(self, chat_id: int, user_id: int) -> Optional[Dict[str, float]]:
        """Returns per-chat and global karma for a user."""
        row_chat = await self.db.execute_query(
            "SELECT rep FROM user_rep_current WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
            fetch_one=True,
        )
        chat_rep = float(getattr(row_chat, "rep", 0.0) or 0.0) if row_chat else 0.0
        row_global = await self.db.execute_query(
            "SELECT SUM(rep) AS total FROM user_rep_current WHERE user_id = ?",
            (user_id,),
            fetch_one=True,
        )
        total = float(getattr(row_global, "total", 0.0) or 0.0) if row_global else 0.0
        return {"received": total, "received_chat": chat_rep}

    async def get_top_users(self, chat_id: int, limit: int = 10) -> List[Dict[str, float]]:
        # Some drivers do not allow parameterizing TOP, so inline the validated integer
        top_n = int(max(1, limit) * 5)
        query = (
            f"SELECT TOP {top_n} urc.user_id, urc.rep, urc.last_seen, up.Username, up.FirstName "
            f"FROM user_rep_current urc WITH (NOLOCK) "
            f"LEFT JOIN UserProfiles up WITH (NOLOCK) ON up.UserID = urc.user_id "
            f"WHERE urc.chat_id = ? "
            f"ORDER BY urc.rep DESC"
        )
        rows = await self.db.execute_query(
            query,
            (chat_id,),
            fetch_all=True,
        )
        if not rows:
            print(f"KARMA: get_top_users -> no rows for chat {chat_id}. Consider running /karmadiag")
            return []
        # Apply exponential decay on-the-fly for display
        from datetime import datetime, timezone
        lam = self._lambda_from_halflife(self.user_halflife_days)
        now = datetime.now(timezone.utc)
        tmp: List[Tuple[int, float, str]] = []
        for r in rows:
            rep = float(getattr(r, "rep", 0.0) or 0.0)
            last_seen = getattr(r, "last_seen", None)
            if last_seen is None:
                decayed = rep
            else:
                dt_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
                decayed = rep * math.exp(-lam * dt_days)
            uid = int(getattr(r, "user_id", 0))
            uname = getattr(r, "Username", None)
            fname = getattr(r, "FirstName", None)
            display = (f"@{uname}" if uname else (fname or str(uid)))
            tmp.append((uid, decayed, display))
        tmp.sort(key=lambda x: x[1], reverse=True)
        top = tmp[:limit]
        return [{"user_id": uid, "name": name, "score": round(score, 2)} for uid, score, name in top]

    async def get_global_top(self, limit: int = 10) -> List[Dict[str, float]]:
        top_n = int(max(1, limit) * 5)
        query = (
            f"SELECT TOP {top_n} urc.user_id, SUM(urc.rep) AS total, MAX(up.Username) AS Username, MAX(up.FirstName) AS FirstName "
            f"FROM user_rep_current urc WITH (NOLOCK) "
            f"LEFT JOIN UserProfiles up WITH (NOLOCK) ON up.UserID = urc.user_id "
            f"GROUP BY urc.user_id ORDER BY SUM(urc.rep) DESC"
        )
        rows = await self.db.execute_query(query, fetch_all=True)
        if not rows:
            print("KARMA: get_global_top -> no rows in user_rep_current. Run /karmadiag for details.")
            return []
        tmp = []
        for r in rows:
            uid = int(getattr(r, "user_id", 0))
            score = float(getattr(r, "total", 0.0) or 0.0)
            uname = getattr(r, "Username", None)
            fname = getattr(r, "FirstName", None)
            name = (f"@{uname}" if uname else (fname or str(uid)))
            tmp.append((uid, score, name))
        tmp.sort(key=lambda x: x[1], reverse=True)
        top = tmp[:limit]
        return [{"user_id": uid, "name": name, "score": round(score, 2)} for uid, score, name in top]
