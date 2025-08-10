# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# enkibot/modules/karma_manager.py
# (Your GPLv3 Header)

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)

KARMA_COOLDOWN_MINUTES = 5

# Regex patterns for simple text-based votes
_THANKS_PATTERN = re.compile(r"^(?:thx|thanks)$", re.IGNORECASE)
_UPVOTE_PATTERN = re.compile(r"^(?:\+|\+\+|\+1|good|great|bravo|well\s*done)$", re.IGNORECASE)
_DOWNVOTE_PATTERN = re.compile(r"^(?:\-|\-\-|\-1|boo|dislike|bad)$", re.IGNORECASE)

# Default emoji/weight mapping from the karma spec
DEFAULT_EMOJI_WEIGHTS: Dict[str, float] = {
    "helpful": 1.0,
    "insightful": 1.2,
    "precise": 1.0,
    "funny": 0.6,
    "thanks": 0.8,
    "low_quality": -0.8,
    "misleading": -1.5,
    "offtopic": -0.5,
}


@dataclass
class KarmaConfig:
    """Minimal representation of per‑chat karma configuration."""
    emoji_map: Dict[str, float] = field(default_factory=lambda: DEFAULT_EMOJI_WEIGHTS.copy())
    decay_msg_days: int = 7
    decay_user_days: int = 45
    allow_downvotes: bool = True
    daily_budget: int = 18
    downvote_quorum: int = 4
    diversity_window_hours: int = 12
    reciprocity_threshold: float = 0.30
    preset: str = "medium"
    auto_tune: bool = True

class KarmaManager:
    def __init__(self, db_manager: 'DatabaseManager'):
        logger.info("KarmaManager initialized.")
        self.db_manager = db_manager
        self.config = KarmaConfig()

    # ------------------------------------------------------------------
    # Parsing & Weight computation
    # ------------------------------------------------------------------
    def parse_vote_token(self, text: str) -> Optional[Tuple[float, str]]:
        """Return (base_weight, tag) for known vote tokens."""
        stripped = text.strip()
        if _THANKS_PATTERN.match(stripped):
            return self.config.emoji_map["thanks"], "thanks"
        if _UPVOTE_PATTERN.match(stripped):
            return self.config.emoji_map["helpful"], "helpful"
        if _DOWNVOTE_PATTERN.match(stripped):
            # Treat all negatives as low_quality for now
            return self.config.emoji_map["low_quality"], "low_quality"
        return None

    async def handle_text_vote(
        self, giver_id: int, receiver_id: int, chat_id: int, message_text: str
    ) -> Optional[str]:
        """Parse a message for karma tokens and apply the change."""
        parsed = self.parse_vote_token(message_text)
        if not parsed:
            return None
        base, tag = parsed
        return await self.change_karma(giver_id, receiver_id, chat_id, base, tag)

    async def _get_rater_trust(self, chat_id: int, user_id: int) -> float:
        """Fetch rater trust from the trust_table if present."""
        query = "SELECT trust FROM trust_table WHERE chat_id = ? AND user_id = ?"
        row = await self.db_manager.execute_query(query, (chat_id, user_id), fetch_one=True)
        if row and getattr(row, "trust", None) is not None:
            try:
                return float(row.trust)
            except Exception:
                pass
        return 1.0

    async def change_karma(
        self,
        giver_id: int,
        receiver_id: int,
        chat_id: int,
        base: float,
        tag: Optional[str] = None,
        msg_id: Optional[int] = None,
    ) -> Optional[str]:
        """Apply a karma vote and persist an event with weights."""
        if giver_id == receiver_id:
            return "self_karma_error"  # Key for language file

        # Cooldown check using legacy KarmaLog for now
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
                logger.info(
                    f"Karma cooldown active for {giver_id} -> {receiver_id}."
                )
                return "cooldown_error"

        # Compute weight components
        rater_trust = await self._get_rater_trust(chat_id, giver_id)
        diversity = 1.0
        anti_collusion = 1.0
        novelty = 1.0
        content_factor = 1.0
        weight = base * rater_trust * diversity * anti_collusion * novelty * content_factor

        # Persist to karma_events table
        event_query = """
            INSERT INTO karma_events (
                chat_id, msg_id, target_user_id, rater_user_id, emoji, base,
                rater_trust, diversity, anti_collusion, novelty, content_factor, weight, ts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,SYSUTCDATETIME())
        """
        params = (
            chat_id,
            msg_id,
            receiver_id,
            giver_id,
            tag,
            base,
            rater_trust,
            diversity,
            anti_collusion,
            novelty,
            content_factor,
            weight,
        )
        await self.db_manager.execute_query(event_query, params, commit=True)

        # Legacy KarmaLog for backwards-compatible leaderboards
        points = 1 if weight > 0 else -1
        log_query = (
            "INSERT INTO KarmaLog (ChatID, GiverUserID, ReceiverUserID, Points) VALUES (?, ?, ?, ?)"
        )
        await self.db_manager.execute_query(
            log_query, (chat_id, giver_id, receiver_id, points), commit=True
        )

        receiver_update_query = (
            "UPDATE UserProfiles SET KarmaReceived = ISNULL(KarmaReceived, 0) + ? WHERE UserID = ?"
        )
        await self.db_manager.execute_query(
            receiver_update_query, (points, receiver_id), commit=True
        )

        if points > 0:
            giver_update_query = (
                "UPDATE UserProfiles SET KarmaGiven = ISNULL(KarmaGiven, 0) + 1 WHERE UserID = ?"
            )
        else:
            giver_update_query = (
                "UPDATE UserProfiles SET HateGiven = ISNULL(HateGiven, 0) + 1 WHERE UserID = ?"
            )
        await self.db_manager.execute_query(giver_update_query, (giver_id,), commit=True)

        logger.info(
            f"Karma changed: {giver_id} -> {receiver_id} weight {weight:.2f} in chat {chat_id}."
        )
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
