# enkibot/modules/stats_manager.py
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
# - Track additional metrics such as hourly activity histograms.
# - Provide graphical representations of statistics.
# - Expand unit tests for edge cases and larger datasets.
# -------------------------------------------------------------------------------
"""Chat statistics tracking and retrieval."""

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)

class StatsManager:
    """Manage chat and user statistics with optional persistence."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.memory_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_messages": 0,
                "joins": 0,
                "leaves": 0,
                "users": defaultdict(
                    lambda: {
                        "count": 0,
                        "first_seen": None,
                        "last_active": None,
                        "username": None,
                    }
                ),
                "links": defaultdict(int),
            }
        )

    async def log_message(
        self, chat_id: int, user_id: int, message_text: Optional[str], username: Optional[str] = None
    ):
        """Increment counters for a new message."""
        if self.db_manager.connection_string:
            await self.db_manager.execute_query(
                """
                MERGE ChatStats AS t
                USING (VALUES (?)) AS s(ChatID)
                ON t.ChatID = s.ChatID
                WHEN MATCHED THEN
                    UPDATE SET TotalMessages = t.TotalMessages + 1, LastUpdated = GETDATE()
                WHEN NOT MATCHED THEN
                    INSERT (ChatID, TotalMessages, JoinCount, LeaveCount, LastUpdated)
                    VALUES (s.ChatID, 1, 0, 0, GETDATE());
                """,
                (chat_id,),
                commit=True,
            )
            await self.db_manager.execute_query(
                """
                MERGE ChatUserStats AS t
                USING (VALUES (?, ?)) AS s(ChatID, UserID)
                ON t.ChatID = s.ChatID AND t.UserID = s.UserID
                WHEN MATCHED THEN
                    UPDATE SET MessageCount = t.MessageCount + 1, LastActive = GETDATE()
                WHEN NOT MATCHED THEN
                    INSERT (ChatID, UserID, MessageCount, FirstSeen, LastActive)
                    VALUES (s.ChatID, s.UserID, 1, GETDATE(), GETDATE());
                """,
                (chat_id, user_id),
                commit=True,
            )
            if message_text:
                for domain in self._extract_domains(message_text):
                    await self.db_manager.execute_query(
                        """
                        MERGE ChatLinkStats AS t
                        USING (VALUES (?, ?)) AS s(ChatID, Domain)
                        ON t.ChatID = s.ChatID AND t.Domain = s.Domain
                        WHEN MATCHED THEN
                            UPDATE SET LinkCount = t.LinkCount + 1
                        WHEN NOT MATCHED THEN
                            INSERT (ChatID, Domain, LinkCount) VALUES (s.ChatID, s.Domain, 1);
                        """,
                        (chat_id, domain),
                        commit=True,
                    )
            return

        stats = self.memory_stats[chat_id]
        stats["total_messages"] += 1
        user_stats = stats["users"][user_id]
        user_stats["count"] += 1
        now = datetime.utcnow()
        user_stats["last_active"] = now
        if username:
            user_stats["username"] = username
        if not user_stats["first_seen"]:
            user_stats["first_seen"] = now
        if message_text:
            for domain in self._extract_domains(message_text):
                stats["links"][domain] += 1

    async def log_member_join(
        self, chat_id: int, user_id: int, username: Optional[str] = None
    ):
        if self.db_manager.connection_string:
            await self.db_manager.execute_query(
                """
                MERGE ChatStats AS t USING (VALUES (?)) AS s(ChatID)
                ON t.ChatID = s.ChatID
                WHEN MATCHED THEN UPDATE SET JoinCount = t.JoinCount + 1, LastUpdated = GETDATE()
                WHEN NOT MATCHED THEN INSERT (ChatID, TotalMessages, JoinCount, LeaveCount, LastUpdated)
                    VALUES (s.ChatID, 0, 1, 0, GETDATE());
                """,
                (chat_id,), commit=True,
            )
            await self.db_manager.execute_query(
                """
                MERGE ChatUserStats AS t USING (VALUES (?, ?)) AS s(ChatID, UserID)
                ON t.ChatID = s.ChatID AND t.UserID = s.UserID
                WHEN NOT MATCHED THEN
                    INSERT (ChatID, UserID, MessageCount, FirstSeen, LastActive)
                    VALUES (s.ChatID, s.UserID, 0, GETDATE(), GETDATE());
                """,
                (chat_id, user_id), commit=True,
            )
            return
        stats = self.memory_stats[chat_id]
        stats["joins"] += 1
        user_stats = stats["users"][user_id]
        now = datetime.utcnow()
        if username:
            user_stats["username"] = username
        if not user_stats["first_seen"]:
            user_stats["first_seen"] = now
            user_stats["last_active"] = now

    async def log_member_leave(self, chat_id: int, user_id: int):
        if self.db_manager.connection_string:
            await self.db_manager.execute_query(
                """
                MERGE ChatStats AS t USING (VALUES (?)) AS s(ChatID)
                ON t.ChatID = s.ChatID
                WHEN MATCHED THEN UPDATE SET LeaveCount = t.LeaveCount + 1, LastUpdated = GETDATE()
                WHEN NOT MATCHED THEN INSERT (ChatID, TotalMessages, JoinCount, LeaveCount, LastUpdated)
                    VALUES (s.ChatID, 0, 0, 1, GETDATE());
                """,
                (chat_id,), commit=True,
            )
            return
        stats = self.memory_stats[chat_id]
        stats["leaves"] += 1

    async def get_chat_stats(self, chat_id: int, top_n: int = 3) -> Optional[Dict[str, Any]]:
        if self.db_manager.connection_string:
            row = await self.db_manager.execute_query(
                "SELECT TotalMessages, JoinCount, LeaveCount FROM ChatStats WHERE ChatID = ?",
                (chat_id,), fetch_one=True,
            )
            if not row:
                return None
            top_users = await self.db_manager.execute_query(
                """
                SELECT TOP (?) cus.UserID, cus.MessageCount, up.Username
                FROM ChatUserStats cus
                LEFT JOIN UserProfiles up ON cus.UserID = up.UserID
                WHERE cus.ChatID = ?
                ORDER BY cus.MessageCount DESC
                """,
                (top_n, chat_id), fetch_all=True,
            )
            top_links = await self.db_manager.execute_query(
                "SELECT TOP (?) Domain, LinkCount FROM ChatLinkStats WHERE ChatID = ? ORDER BY LinkCount DESC",
                (top_n, chat_id), fetch_all=True,
            )
            return {
                "total_messages": row.TotalMessages,
                "joins": row.JoinCount,
                "leaves": row.LeaveCount,
                "top_users": [
                    {
                        "user_id": r.UserID,
                        "username": r.Username,
                        "count": r.MessageCount,
                    }
                    for r in top_users
                ] if top_users else [],
                "top_links": [
                    {"domain": r.Domain, "count": r.LinkCount} for r in top_links
                ] if top_links else [],
            }

        stats = self.memory_stats.get(chat_id)
        if not stats:
            return None
        top_users_sorted = sorted(
            (
                {
                    "user_id": uid,
                    "username": info.get("username"),
                    "count": info["count"],
                }
                for uid, info in stats["users"].items()
            ),
            key=lambda x: x["count"],
            reverse=True,
        )[:top_n]
        top_links_sorted = sorted(
            (
                {"domain": d, "count": c} for d, c in stats["links"].items()
            ),
            key=lambda x: x["count"],
            reverse=True,
        )[:top_n]
        return {
            "total_messages": stats["total_messages"],
            "joins": stats["joins"],
            "leaves": stats["leaves"],
            "top_users": top_users_sorted,
            "top_links": top_links_sorted,
        }

    async def get_user_stats(self, chat_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        if self.db_manager.connection_string:
            user_row = await self.db_manager.execute_query(
                """
                SELECT cus.MessageCount, cus.FirstSeen, cus.LastActive, up.Username
                FROM ChatUserStats cus
                LEFT JOIN UserProfiles up ON cus.UserID = up.UserID
                WHERE cus.ChatID = ? AND cus.UserID = ?
                """,
                (chat_id, user_id), fetch_one=True,
            )
            if not user_row:
                return None
            total_row = await self.db_manager.execute_query(
                "SELECT TotalMessages FROM ChatStats WHERE ChatID = ?",
                (chat_id,), fetch_one=True,
            )
            rank_row = await self.db_manager.execute_query(
                "SELECT COUNT(*) + 1 AS Rank FROM ChatUserStats WHERE ChatID = ? AND MessageCount > ?",
                (chat_id, user_row.MessageCount), fetch_one=True,
            )
            total_users_row = await self.db_manager.execute_query(
                "SELECT COUNT(*) AS Cnt FROM ChatUserStats WHERE ChatID = ?",
                (chat_id,), fetch_one=True,
            )
            return {
                "messages": user_row.MessageCount,
                "first_seen": user_row.FirstSeen,
                "last_active": user_row.LastActive,
                "username": user_row.Username,
                "total_messages": total_row.TotalMessages if total_row else 0,
                "rank": rank_row.Rank if rank_row else 1,
                "total_users": total_users_row.Cnt if total_users_row else 1,
            }

        stats = self.memory_stats.get(chat_id)
        if not stats or user_id not in stats["users"]:
            return None
        user_stats = stats["users"][user_id]
        return {
            "messages": user_stats["count"],
            "first_seen": user_stats["first_seen"],
            "last_active": user_stats["last_active"],
            "username": user_stats.get("username"),
            "total_messages": stats["total_messages"],
            "rank": 1 + sum(1 for u in stats["users"].values() if u["count"] > user_stats["count"]),
            "total_users": len(stats["users"]),
        }

    def _extract_domains(self, text: str) -> List[str]:
        urls = re.findall(r"https?://[^\s]+", text)
        domains = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    domains.append(parsed.netloc.lower())
            except Exception:
                continue
        return domains
