# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# enkibot/modules/karma_manager.py
# (Your GPLv3 Header)

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)

KARMA_COOLDOWN_MINUTES = 5

class KarmaManager:
    def __init__(self, db_manager: 'DatabaseManager'):
        logger.info("KarmaManager initialized.")
        self.db_manager = db_manager

    async def change_karma(self, giver_id: int, receiver_id: int, chat_id: int, points: int) -> Optional[str]:
        """
        Changes a user's karma, handles cooldowns, and updates totals.
        Returns a status message or None if no action is taken.
        """
        if giver_id == receiver_id:
            return "self_karma_error" # Key for language file

        # Check cooldown
        cooldown_check_query = """
            SELECT TOP 1 Timestamp FROM KarmaLog
            WHERE GiverUserID = ? AND ReceiverUserID = ? AND ChatID = ?
            ORDER BY Timestamp DESC
        """
        last_karma_time_row = await self.db_manager.execute_query(
            cooldown_check_query, (giver_id, receiver_id, chat_id), fetch_one=True
        )

        if last_karma_time_row and last_karma_time_row[0]:
            last_karma_time = last_karma_time_row[0]
            if datetime.utcnow() < last_karma_time + timedelta(minutes=KARMA_COOLDOWN_MINUTES):
                logger.info(f"Karma cooldown active for {giver_id} -> {receiver_id}.")
                return "cooldown_error"

        # Proceed to change karma
        log_query = "INSERT INTO KarmaLog (ChatID, GiverUserID, ReceiverUserID, Points) VALUES (?, ?, ?, ?)"
        await self.db_manager.execute_query(log_query, (chat_id, giver_id, receiver_id, points), commit=True)
        
        # Update receiver's total
        receiver_update_query = "UPDATE UserProfiles SET KarmaReceived = ISNULL(KarmaReceived, 0) + ? WHERE UserID = ?"
        await self.db_manager.execute_query(receiver_update_query, (points, receiver_id), commit=True)

        # Update giver's stats
        if points > 0:
            giver_update_query = "UPDATE UserProfiles SET KarmaGiven = ISNULL(KarmaGiven, 0) + 1 WHERE UserID = ?"
        else:
            giver_update_query = "UPDATE UserProfiles SET HateGiven = ISNULL(HateGiven, 0) + 1 WHERE UserID = ?"
        await self.db_manager.execute_query(giver_update_query, (giver_id,), commit=True)

        logger.info(f"Karma changed: {giver_id} gave {points} points to {receiver_id} in chat {chat_id}.")
        return "karma_changed_success"

    async def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Fetches karma stats for a single user."""
        query = "SELECT FirstName, KarmaReceived, KarmaGiven, HateGiven FROM UserProfiles WHERE UserID = ?"
        row = await self.db_manager.execute_query(query, (user_id,), fetch_one=True)
        if row:
            return {
                "name": row.FirstName,
                "received": row.KarmaReceived or 0,
                "given": row.KarmaGiven or 0,
                "hate": row.HateGiven or 0
            }
        return None

    async def get_top_users(self, chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Gets the karma leaderboard for a specific chat."""
        # This query joins UserProfiles with ChatLog to find users active in the specific chat.
        query = f"""
            SELECT TOP (?) up.FirstName, up.KarmaReceived
            FROM UserProfiles up
            WHERE up.UserID IN (SELECT DISTINCT UserID FROM ChatLog WHERE ChatID = ?)
            ORDER BY up.KarmaReceived DESC
        """
        rows = await self.db_manager.execute_query(query, (limit, chat_id), fetch_all=True)
        return [{"name": row.FirstName, "score": row.KarmaReceived or 0} for row in rows] if rows else []

    async def get_top_givers(self, chat_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Gets the top karma givers leaderboard for a specific chat."""
        query = f"""
            SELECT TOP (?) up.FirstName, up.KarmaGiven, up.HateGiven
            FROM UserProfiles up
            WHERE up.UserID IN (SELECT DISTINCT UserID FROM ChatLog WHERE ChatID = ?)
            ORDER BY (up.KarmaGiven + up.HateGiven) DESC
        """
        rows = await self.db_manager.execute_query(query, (limit, chat_id), fetch_all=True)
        return [{"name": row.FirstName, "given": row.KarmaGiven or 0, "hate": row.HateGiven or 0} for row in rows] if rows else []
