# enkibot/utils/database.py
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
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Ensure your GPLv3 header is here)

# <<<--- DIAGNOSTIC PRINT IR-1: VERY TOP OF INTENT_RECOGNIZER.PY --- >>>
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)
# ==================================================================================================
# === EnkiBot Database Utilities ===
# ==================================================================================================
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import date
from types import SimpleNamespace

try:  # pragma: no cover - optional dependency
    import pyodbc
except Exception:  # pragma: no cover
    pyodbc = SimpleNamespace(Error=Exception)

from enkibot import config  # For DB_CONNECTION_STRING

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.connection_string = connection_string
        if not self.connection_string:
            logger.warning(
                "Database connection string is not configured. Database operations will be disabled."
            )

    def _log_db_error(
        self, context: str, query: str, params: Optional[tuple], err: Exception
    ) -> None:
        msg = str(err)
        lower = msg.lower()
        hint = ""
        if "no such table" in lower or "does not exist" in lower:
            hint = " Possible missing table."
        elif "no such column" in lower or "unknown column" in lower:
            hint = " Possible missing column."
        logger.error(
            f"{context} DB error on '{query[:100]}...' with params {params}: {msg}{hint}",
            exc_info=True,
        )

    def get_db_connection(self) -> Optional[Any]:
        if not self.connection_string or not hasattr(pyodbc, "connect"):
            logger.warning(
                "Database connection unavailable: missing connection string or pyodbc."
            )
            return None
        try:
            conn = pyodbc.connect(self.connection_string, autocommit=False)
            logger.debug("Database connection established.")
            return conn
        except pyodbc.Error as ex:
            logger.error(
                f"Database connection error: {ex.args[0] if ex.args else ''} - {ex}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error establishing database connection: {e}", exc_info=True
            )
            return None

    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False,
    ):
        if not self.connection_string:
            logger.warning("Query execution skipped: Database not configured.")
            return None if fetch_one or fetch_all else (True if commit else False)
        conn = self.get_db_connection()
        if not conn:
            return None if fetch_one or fetch_all else (True if commit else False)
        try:
            with conn.cursor() as cursor:
                logger.debug(f"Executing query: {query[:150]}... with params: {params}")
                cursor.execute(query, params) if params else cursor.execute(query)
                if commit:
                    conn.commit()
                    logger.debug("Query committed.")
                    return True
                if fetch_one:
                    row = cursor.fetchone()
                    logger.debug(f"Query fetch_one: {row}")
                    return row
                if fetch_all:
                    rows = cursor.fetchall()
                    logger.debug(f"Query fetch_all count: {len(rows)}")
                    return rows
            return True
        except pyodbc.Error as ex:
            self._log_db_error("Query", query, params, ex)
            try:
                conn.rollback()
                logger.info("Transaction rolled back.")
            except pyodbc.Error as rb_ex:
                logger.error(f"Error during rollback: {rb_ex}", exc_info=True)
            return None if fetch_one or fetch_all else False
        except Exception as e:
            self._log_db_error("Query", query, params, e)
            return None if fetch_one or fetch_all else False
        finally:
            if conn:
                conn.close()
                logger.debug("DB connection closed post-exec.")

    async def get_recent_chat_texts(self, chat_id: int, limit: int = 3) -> List[str]:
        if not self.connection_string:
            return []
        query = """
            SELECT TOP (?) MessageText FROM ChatLog
            WHERE ChatID = ? AND MessageText IS NOT NULL AND RTRIM(LTRIM(MessageText)) != ''
            ORDER BY Timestamp DESC
        """
        actual_limit = max(1, limit)
        rows = await self.execute_query(query, (actual_limit, chat_id), fetch_all=True)
        return (
            [row.MessageText for row in reversed(rows) if row.MessageText]
            if rows
            else []
        )

    async def log_chat_message_and_upsert_user(
        self,
        chat_id: int,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        last_name: Optional[str],
        message_id: int,
        message_text: str,
        preferred_language: Optional[str] = None,
    ) -> Optional[str]:
        if not self.connection_string:
            return None
        upsert_user_sql = """
            MERGE UserProfiles AS t
            USING (VALUES(?,?,?,?,GETDATE(),?)) AS s(UserID,Username,FirstName,LastName,LastSeen,PreferredLanguage)
            ON t.UserID = s.UserID
            WHEN MATCHED THEN
                UPDATE SET Username=s.Username, FirstName=s.FirstName, LastName=s.LastName, LastSeen=s.LastSeen, MessageCount=ISNULL(t.MessageCount,0)+1, PreferredLanguage=ISNULL(s.PreferredLanguage, t.PreferredLanguage)
            WHEN NOT MATCHED THEN
                INSERT(UserID,Username,FirstName,LastName,LastSeen,MessageCount,ProfileLastUpdated, PreferredLanguage)
                VALUES(s.UserID,s.Username,s.FirstName,s.LastName,s.LastSeen,1,GETDATE(),s.PreferredLanguage)
            OUTPUT $action AS Action;
        """
        user_params = (user_id, username, first_name, last_name, preferred_language)
        chat_log_sql = "INSERT INTO ChatLog (ChatID, UserID, Username, FirstName, MessageID, MessageText, Timestamp) VALUES (?, ?, ?, ?, ?, ?, GETDATE())"
        chat_log_params = (
            chat_id,
            user_id,
            username,
            first_name,
            message_id,
            message_text,
        )
        conn = self.get_db_connection()
        if not conn:
            return None
        action_taken = None
        try:
            with conn.cursor() as cursor:
                cursor.execute(upsert_user_sql, user_params)
                row = cursor.fetchone()
                if row:
                    action_taken = row.Action
                cursor.execute(chat_log_sql, chat_log_params)
            conn.commit()
            logger.info(
                f"Logged message for user {user_id}. Profile action: {action_taken}"
            )
            return action_taken
        except pyodbc.Error as ex:
            self._log_db_error(
                "log_chat_message_and_upsert_user",
                upsert_user_sql,
                user_params,
                ex,
            )
            conn.rollback()
        finally:
            if conn:
                conn.close()
        return None  # Ensure a return path if try fails before commit

    async def log_assistant_invocation(
        self,
        chat_id: int,
        user_id: int,
        message_id: int,
        detected: bool,
        alias: Optional[str],
        prompt: str,
        reason: Optional[str],
        lang: Optional[str],
        routed_to_llm: bool,
        llm_ok: bool,
        error: Optional[str],
    ) -> None:
        if not self.connection_string:
            return
        sql = (
            "INSERT INTO assistant_invocations (chat_id, user_id, message_id, detected, alias, prompt, reason, lang, routed_to_llm, llm_ok, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            chat_id,
            user_id,
            message_id,
            int(detected),
            alias,
            prompt,
            reason,
            lang,
            int(routed_to_llm),
            int(llm_ok),
            error,
        )
        await self.execute_query(sql, params, commit=True)

    async def get_conversation_history(
        self, chat_id: int, limit: int = 20
    ) -> List[Dict[str, str]]:
        query = "SELECT TOP (?) Role, Content FROM ConversationHistory WHERE ChatID = ? ORDER BY Timestamp DESC"
        rows = await self.execute_query(query, (limit, chat_id), fetch_all=True)
        return (
            [
                {"role": row.Role.lower(), "content": row.Content}
                for row in reversed(rows)
            ]
            if rows
            else []
        )

    async def save_to_conversation_history(
        self,
        chat_id: int,
        entity_id: int,
        message_id_telegram: Optional[int],
        role: str,
        content: str,
    ):
        query = "INSERT INTO ConversationHistory (ChatID, UserID, MessageID, Role, Content, Timestamp) VALUES (?, ?, ?, ?, ?, GETDATE())"
        await self.execute_query(
            query, (chat_id, entity_id, message_id_telegram, role, content), commit=True
        )

    async def get_user_profile_notes(self, user_id: int) -> Optional[str]:
        row = await self.execute_query(
            "SELECT Notes FROM UserProfiles WHERE UserID = ?",
            (user_id,),
            fetch_one=True,
        )
        return row.Notes if row and row.Notes else None

    async def update_user_profile_notes(self, user_id: int, notes: str):
        await self.execute_query(
            "UPDATE UserProfiles SET Notes = ?, ProfileLastUpdated = GETDATE() WHERE UserID = ?",
            (notes, user_id),
            commit=True,
        )
        logger.info(f"Profile notes updated for user {user_id}.")

    async def save_user_name_variations(self, user_id: int, variations: List[str]):
        if not self.connection_string or not variations:
            return
        sql_merge = """
            MERGE INTO UserNameVariations AS t USING (SELECT ? AS UserID, ? AS NameVariation) AS s
            ON (t.UserID = s.UserID AND t.NameVariation = s.NameVariation)
            WHEN NOT MATCHED THEN INSERT (UserID, NameVariation) VALUES (s.UserID, s.NameVariation);
        """
        conn = self.get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cursor:
                params_to_insert = [
                    (user_id, var) for var in variations if var and str(var).strip()
                ]
                if params_to_insert:
                    cursor.executemany(sql_merge, params_to_insert)
            conn.commit()
            logger.info(
                f"Saved/updated {len(params_to_insert)} name vars for user {user_id}."
            )
        except pyodbc.Error as ex:
            logger.error(
                f"DB error saving name vars for user {user_id}: {ex}", exc_info=True
            )
            conn.rollback()
        finally:
            if conn:
                conn.close()

    async def find_user_profiles_by_name_variation(
        self, name_variation_query: str
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT DISTINCT up.UserID, up.FirstName, up.LastName, up.Username, up.Notes
            FROM UserProfiles up JOIN UserNameVariations unv ON up.UserID = unv.UserID
            WHERE unv.NameVariation = ?
        """
        rows = await self.execute_query(
            query, (name_variation_query.lower(),), fetch_all=True
        )
        return (
            [
                {
                    "UserID": r.UserID,
                    "FirstName": r.FirstName,
                    "LastName": r.LastName,
                    "Username": r.Username,
                    "Notes": r.Notes,
                }
                for r in rows
            ]
            if rows
            else []
        )

    async def get_user_messages_from_chat_log(
        self, user_id: int, chat_id: int, limit: int = 10
    ) -> List[str]:  # Kept from previous version
        query = "SELECT TOP (?) MessageText FROM ChatLog WHERE UserID = ? AND ChatID = ? AND MessageText IS NOT NULL AND RTRIM(LTRIM(MessageText)) != '' ORDER BY Timestamp DESC"
        rows = await self.execute_query(
            query, (limit, user_id, chat_id), fetch_all=True
        )
        return [row.MessageText for row in rows] if rows else []

    async def _ensure_user_usage_table(self):
        """Create the UserUsage table if it does not exist."""
        await self.execute_query(
            """
            IF OBJECT_ID('UserUsage', 'U') IS NULL
            CREATE TABLE UserUsage (
                UserID BIGINT NOT NULL,
                UsageDate DATE NOT NULL,
                LlmCount INT NOT NULL DEFAULT 0,
                ImageCount INT NOT NULL DEFAULT 0,
                CONSTRAINT PK_UserUsage PRIMARY KEY (UserID, UsageDate)
            );
            """,
            commit=True,
        )

    async def get_daily_usage(
        self, user_id: int, usage_date: Optional[date] = None
    ) -> Dict[str, int]:
        usage_date = usage_date or date.today()
        await self._ensure_user_usage_table()
        row = await self.execute_query(
            "SELECT LlmCount, ImageCount FROM UserUsage WHERE UserID = ? AND UsageDate = ?",
            (user_id, usage_date),
            fetch_one=True,
        )
        if row:
            return {"llm": row.LlmCount or 0, "image": row.ImageCount or 0}
        return {"llm": 0, "image": 0}

    async def increment_usage(
        self, user_id: int, usage_type: str, usage_date: Optional[date] = None
    ):
        usage_date = usage_date or date.today()
        column = "LlmCount" if usage_type == "llm" else "ImageCount"
        await self._ensure_user_usage_table()
        sql = f"""
            MERGE UserUsage AS t USING (VALUES(?,?)) AS s(UserID,UsageDate)
            ON t.UserID = s.UserID AND t.UsageDate = s.UsageDate
            WHEN MATCHED THEN UPDATE SET {column} = t.{column} + 1
            WHEN NOT MATCHED THEN INSERT (UserID, UsageDate, {column}) VALUES (s.UserID, s.UsageDate, 1);
        """
        await self.execute_query(sql, (user_id, usage_date), commit=True)

    async def check_and_increment_usage(
        self,
        user_id: int,
        usage_type: str,
        daily_limit: int,
        usage_date: Optional[date] = None,
    ) -> bool:
        usage_date = usage_date or date.today()
        usage = await self.get_daily_usage(user_id, usage_date)
        if usage.get(usage_type, 0) >= daily_limit:
            return False
        await self.increment_usage(user_id, usage_type, usage_date)
        return True

    async def add_verified_user(self, user_id: int):
        """Add or update a user in the global verification table."""
        await self.execute_query(
            "MERGE VerifiedUsers AS t USING (VALUES(?)) AS s(UserID) "
            "ON t.UserID = s.UserID "
            "WHEN MATCHED THEN UPDATE SET VerifiedAt = GETDATE() "
            "WHEN NOT MATCHED THEN INSERT (UserID, VerifiedAt) VALUES (s.UserID, GETDATE());",
            (user_id,),
            commit=True,
        )

    async def is_user_verified(self, user_id: int) -> bool:
        row = await self.execute_query(
            "SELECT 1 FROM VerifiedUsers WHERE UserID = ?",
            (user_id,),
            fetch_one=True,
        )
        return bool(row)

    async def add_spam_vote(
        self, chat_id: int, target_user_id: int, reporter_user_id: int
    ) -> bool:
        """Records a spam vote if one doesn't already exist for this reporter/target pair."""
        if not self.connection_string:
            return False
        conn = self.get_db_connection()
        if not conn:
            return False
        sql = (
            "MERGE SpamReports AS t USING (VALUES(?,?,?)) AS s(ChatID,TargetUserID,ReporterUserID) "
            "ON t.ChatID=s.ChatID AND t.TargetUserID=s.TargetUserID AND t.ReporterUserID=s.ReporterUserID "
            "WHEN NOT MATCHED THEN INSERT (ChatID,TargetUserID,ReporterUserID,Timestamp) "
            "VALUES(s.ChatID,s.TargetUserID,s.ReporterUserID,GETDATE()) OUTPUT $action;"
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (chat_id, target_user_id, reporter_user_id))
                row = cursor.fetchone()
            conn.commit()
            return bool(row and row.Action == "INSERT")
        except pyodbc.Error as ex:
            logger.error(f"DB error adding spam vote: {ex}", exc_info=True)
            try:
                conn.rollback()
            except pyodbc.Error:
                pass
            return False
        finally:
            if conn:
                conn.close()

    async def count_spam_votes(
        self, chat_id: int, target_user_id: int, window_minutes: int
    ) -> int:
        """Returns the number of unique reporters within the given timeframe."""
        query = (
            "SELECT COUNT(DISTINCT ReporterUserID) AS VoteCount FROM SpamReports "
            "WHERE ChatID = ? AND TargetUserID = ? AND Timestamp > DATEADD(MINUTE, ?, GETDATE())"
        )
        row = await self.execute_query(
            query, (chat_id, target_user_id, -abs(window_minutes)), fetch_one=True
        )
        return row.VoteCount if row and hasattr(row, "VoteCount") else 0

    async def clear_spam_votes(self, chat_id: int, target_user_id: int):
        await self.execute_query(
            "DELETE FROM SpamReports WHERE ChatID = ? AND TargetUserID = ?",
            (chat_id, target_user_id),
            commit=True,
        )

    async def get_spam_vote_threshold(
        self, chat_id: int, default_threshold: int
    ) -> int:
        row = await self.execute_query(
            "SELECT SpamVoteThreshold FROM ChatSettings WHERE ChatID = ?",
            (chat_id,),
            fetch_one=True,
        )
        if row and hasattr(row, "SpamVoteThreshold") and row.SpamVoteThreshold:
            return row.SpamVoteThreshold
        await self.execute_query(
            "INSERT INTO ChatSettings (ChatID, SpamVoteThreshold) VALUES (?, ?)",
            (chat_id, default_threshold),
            commit=True,
        )
        return default_threshold

    async def set_spam_vote_threshold(self, chat_id: int, threshold: int):
        await self.execute_query(
            "MERGE ChatSettings AS t USING (VALUES(?,?)) AS s(ChatID,SpamVoteThreshold) "
            "ON t.ChatID=s.ChatID "
            "WHEN MATCHED THEN UPDATE SET SpamVoteThreshold=s.SpamVoteThreshold "
            "WHEN NOT MATCHED THEN INSERT (ChatID,SpamVoteThreshold) VALUES (s.ChatID,s.SpamVoteThreshold);",
            (chat_id, threshold),
            commit=True,
        )

    async def _ensure_nsfw_columns(self):
        """Ensure NSFW-related columns exist in ChatSettings."""
        await self.execute_query(
            "IF COL_LENGTH('ChatSettings', 'NSFWFilterEnabled') IS NULL "
            "ALTER TABLE ChatSettings ADD NSFWFilterEnabled BIT NOT NULL DEFAULT 0;",
            commit=True,
        )
        await self.execute_query(
            f"IF COL_LENGTH('ChatSettings', 'NSFWThreshold') IS NULL "
            f"ALTER TABLE ChatSettings ADD NSFWThreshold FLOAT NOT NULL DEFAULT {config.NSFW_DETECTION_THRESHOLD};",
            commit=True,
        )

    async def get_nsfw_filter_enabled(self, chat_id: int) -> bool:
        await self._ensure_nsfw_columns()
        row = await self.execute_query(
            "SELECT NSFWFilterEnabled FROM ChatSettings WHERE ChatID = ?",
            (chat_id,),
            fetch_one=True,
        )
        if row and hasattr(row, "NSFWFilterEnabled"):
            return bool(row.NSFWFilterEnabled)
        await self.execute_query(
            "INSERT INTO ChatSettings (ChatID, SpamVoteThreshold, NSFWFilterEnabled, NSFWThreshold) VALUES (?, ?, ?, ?)",
            (
                chat_id,
                config.DEFAULT_SPAM_VOTE_THRESHOLD,
                int(config.NSFW_FILTER_DEFAULT_ENABLED),
                config.NSFW_DETECTION_THRESHOLD,
            ),
            commit=True,
        )
        return bool(config.NSFW_FILTER_DEFAULT_ENABLED)

    async def set_nsfw_filter_enabled(self, chat_id: int, enabled: bool):
        await self._ensure_nsfw_columns()
        await self.execute_query(
            "MERGE ChatSettings AS t USING (VALUES(?,?,?,?)) AS s(ChatID,SpamVoteThreshold,NSFWFilterEnabled,NSFWThreshold) "
            "ON t.ChatID=s.ChatID "
            "WHEN MATCHED THEN UPDATE SET NSFWFilterEnabled=s.NSFWFilterEnabled "
            "WHEN NOT MATCHED THEN INSERT (ChatID,SpamVoteThreshold,NSFWFilterEnabled,NSFWThreshold) VALUES (s.ChatID,s.SpamVoteThreshold,s.NSFWFilterEnabled,s.NSFWThreshold);",
            (
                chat_id,
                config.DEFAULT_SPAM_VOTE_THRESHOLD,
                int(enabled),
                config.NSFW_DETECTION_THRESHOLD,
            ),
            commit=True,
        )

    async def get_nsfw_threshold(self, chat_id: int) -> float:
        await self._ensure_nsfw_columns()
        row = await self.execute_query(
            "SELECT NSFWThreshold FROM ChatSettings WHERE ChatID = ?",
            (chat_id,),
            fetch_one=True,
        )
        if row and hasattr(row, "NSFWThreshold") and row.NSFWThreshold is not None:
            return float(row.NSFWThreshold)
        await self.execute_query(
            "INSERT INTO ChatSettings (ChatID, SpamVoteThreshold, NSFWFilterEnabled, NSFWThreshold) VALUES (?, ?, ?, ?)",
            (
                chat_id,
                config.DEFAULT_SPAM_VOTE_THRESHOLD,
                int(config.NSFW_FILTER_DEFAULT_ENABLED),
                config.NSFW_DETECTION_THRESHOLD,
            ),
            commit=True,
        )
        return float(config.NSFW_DETECTION_THRESHOLD)

    async def set_nsfw_threshold(self, chat_id: int, threshold: float):
        await self._ensure_nsfw_columns()
        await self.execute_query(
            "MERGE ChatSettings AS t USING (VALUES(?,?,?,?)) AS s(ChatID,SpamVoteThreshold,NSFWFilterEnabled,NSFWThreshold) "
            "ON t.ChatID=s.ChatID "
            "WHEN MATCHED THEN UPDATE SET NSFWThreshold=s.NSFWThreshold "
            "WHEN NOT MATCHED THEN INSERT (ChatID,SpamVoteThreshold,NSFWFilterEnabled,NSFWThreshold) VALUES (s.ChatID,s.SpamVoteThreshold,s.NSFWFilterEnabled,s.NSFWThreshold);",
            (
                chat_id,
                config.DEFAULT_SPAM_VOTE_THRESHOLD,
                int(config.NSFW_FILTER_DEFAULT_ENABLED),
                threshold,
            ),
            commit=True,
        )

    async def add_warning(
        self, chat_id: int, user_id: int, reason: Optional[str] = None
    ) -> int:
        """Increment warning count for a user and return new count."""
        if not self.connection_string:
            return 0
        conn = self.get_db_connection()
        if not conn:
            return 0
        sql = (
            "MERGE UserWarnings AS t USING (VALUES(?,?,?)) AS s(ChatID,UserID,Reason) "
            "ON t.ChatID=s.ChatID AND t.UserID=s.UserID "
            "WHEN MATCHED THEN UPDATE SET WarnCount=ISNULL(t.WarnCount,0)+1, LastReason=s.Reason, LastWarned=GETDATE() "
            "WHEN NOT MATCHED THEN INSERT (ChatID,UserID,WarnCount,LastReason,LastWarned) "
            "VALUES (s.ChatID,s.UserID,1,s.Reason,GETDATE()) OUTPUT inserted.WarnCount;"
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (chat_id, user_id, reason))
                row = cursor.fetchone()
            conn.commit()
            return row.WarnCount if row and hasattr(row, "WarnCount") else 0
        except pyodbc.Error as ex:
            logger.error(f"DB error adding warning: {ex}", exc_info=True)
            try:
                conn.rollback()
            except pyodbc.Error:
                pass
            return 0
        finally:
            if conn:
                conn.close()

    async def get_warning_count(self, chat_id: int, user_id: int) -> int:
        row = await self.execute_query(
            "SELECT WarnCount FROM UserWarnings WHERE ChatID = ? AND UserID = ?",
            (chat_id, user_id),
            fetch_one=True,
        )
        return row.WarnCount if row and hasattr(row, "WarnCount") else 0

    async def clear_warnings(self, chat_id: int, user_id: int):
        await self.execute_query(
            "DELETE FROM UserWarnings WHERE ChatID = ? AND UserID = ?",
            (chat_id, user_id),
            commit=True,
        )

    async def list_warnings(self, chat_id: int):
        rows = await self.execute_query(
            "SELECT UserID, WarnCount FROM UserWarnings WHERE ChatID = ?",
            (chat_id,),
            fetch_all=True,
        )
        return [(r.UserID, r.WarnCount) for r in rows] if rows else []

    async def log_moderation_action(
        self,
        chat_id: int,
        user_id: Optional[int],
        message_id: int,
        categories: Optional[str],
    ):
        """Persist a moderation action to the database for audit purposes."""
        await self.execute_query(
            "INSERT INTO ModerationLog (ChatID, UserID, MessageID, Categories) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, message_id, categories),
            commit=True,
        )

    async def log_fact_check(
        self,
        chat_id: int,
        message_id: int,
        claim_text: str,
        verdict: str,
        confidence: float,
        track: str = "news",
        details: Optional[str] = None,
    ):
        """Persist fact-check outcomes for auditing.

        Handles optional ``Track`` and ``Details`` columns for backward
        compatibility.
        """

        columns = ["ChatID", "MessageID", "ClaimText", "Verdict", "Confidence"]
        params: List[Any] = [chat_id, message_id, claim_text, verdict, confidence]

        track_exists = await self.execute_query(
            "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'FactCheckLog' AND COLUMN_NAME = 'Track'",
            fetch_one=True,
        )
        if track_exists:
            columns.append("Track")
            params.append(track)

        details_exists = await self.execute_query(
            "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'FactCheckLog' AND COLUMN_NAME = 'Details'",
            fetch_one=True,
        )
        if details_exists:
            columns.append("Details")
            params.append(details)

        placeholders = ", ".join(["?"] * len(params))
        query = (
            f"INSERT INTO FactCheckLog ({', '.join(columns)}) VALUES ({placeholders})"
        )

        await self.execute_query(query, tuple(params), commit=True)

    async def log_fact_gate(
        self, chat_id: int, message_id: int, p_news: float, p_book: float
    ) -> None:
        await self.execute_query(
            "INSERT INTO FactGateLog (ChatID, MessageID, PNews, PBook) VALUES (?, ?, ?, ?)",
            (chat_id, message_id, p_news, p_book),
            commit=True,
        )

    async def log_web_request(
        self,
        url: str,
        method: str,
        status_code: Optional[int],
        duration_ms: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        """Record an outbound HTTP request for auditing and diagnostics."""
        await self.execute_query(
            "INSERT INTO WebRequestLog (Url, Method, StatusCode, DurationMs, Error) VALUES (?, ?, ?, ?, ?)",
            (url, method, status_code, duration_ms, error),
            commit=True,
        )

    async def log_answer_evidence(
        self,
        chat_id: int,
        asked_by: int,
        intent: str,
        query_text: Optional[str],
        lang: Optional[str],
        items: List[Dict[str, Any]],
    ) -> Optional[int]:
        """Persist evidence snippets for a given answer.

        Returns the generated answer_id if successful.
        """
        if not self.connection_string:
            return None
        conn = self.get_db_connection()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    (
                        "INSERT INTO answer_evidence (chat_id, asked_by, intent, query_text, lang) "
                        "VALUES (?, ?, ?, ?, ?); SELECT SCOPE_IDENTITY();"
                    ),
                    (chat_id, asked_by, intent, query_text, lang),
                )
                row = cursor.fetchone()
                if not row:
                    conn.rollback()
                    return None
                answer_id = int(row[0])
                if items:
                    params = [
                        (
                            answer_id,
                            it.get("message_id"),
                            it.get("rank", 0),
                            it.get("snippet"),
                            it.get("reason"),
                        )
                        for it in items
                    ]
                    cursor.executemany(
                        "INSERT INTO answer_evidence_items (answer_id, message_id, rank, snippet, reason)"
                        " VALUES (?, ?, ?, ?, ?)",
                        params,
                    )
            conn.commit()
            return answer_id
        except pyodbc.Error as ex:
            logger.error(f"DB error logging answer evidence: {ex}", exc_info=True)
            try:
                conn.rollback()
            except pyodbc.Error:
                pass
            return None
        finally:
            if conn:
                conn.close()

    async def save_user_persona_version(
        self,
        chat_id: int,
        user_id: int,
        portrait_md: str,
        traits_json: str,
        signals_json: Optional[str] = None,
    ) -> bool:
        """Store a new persona version for the given user.

        Automatically increments the version number for that chat/user pair.
        """
        if not self.connection_string:
            return False
        conn = self.get_db_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT ISNULL(MAX(version),0) + 1 AS next_version FROM user_persona_versions WHERE chat_id = ? AND user_id = ?",
                    (chat_id, user_id),
                )
                row = cursor.fetchone()
                version = int(
                    row.next_version if row and hasattr(row, "next_version") else 1
                )
                cursor.execute(
                    "INSERT INTO user_persona_versions (chat_id, user_id, version, portrait_md, traits_json, signals_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chat_id, user_id, version, portrait_md, traits_json, signals_json),
                )
            conn.commit()
            return True
        except pyodbc.Error as ex:
            logger.error(f"DB error saving persona version: {ex}", exc_info=True)
            try:
                conn.rollback()
            except pyodbc.Error:
                pass
            return False
        finally:
            if conn:
                conn.close()

    async def get_latest_user_persona(
        self, chat_id: int, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent persona version for a user."""
        row = await self.execute_query(
            "SELECT TOP 1 version, portrait_md, traits_json, signals_json "
            "FROM user_persona_versions WHERE chat_id = ? AND user_id = ? ORDER BY version DESC",
            (chat_id, user_id),
            fetch_one=True,
        )
        if not row:
            return None
        return {
            "version": row.version if hasattr(row, "version") else None,
            "portrait_md": row.portrait_md if hasattr(row, "portrait_md") else None,
            "traits_json": row.traits_json if hasattr(row, "traits_json") else None,
            "signals_json": row.signals_json if hasattr(row, "signals_json") else None,
        }


def initialize_database():  # This function defines and uses DatabaseManager locally
    if not config.DB_CONNECTION_STRING:
        logger.warning("Cannot initialize database: Connection string not configured.")
        return

    db_mngr = DatabaseManager(config.DB_CONNECTION_STRING)
    conn = db_mngr.get_db_connection()
    if not conn:
        logger.error(
            "Failed to connect to database for initialization (initialize_database)."
        )
        return

    conn.autocommit = True
    table_queries = {
        "UserProfiles": "CREATE TABLE UserProfiles (UserID BIGINT PRIMARY KEY, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, LastName NVARCHAR(255) NULL, LastSeen DATETIME2 DEFAULT GETDATE(), MessageCount INT DEFAULT 0, PreferredLanguage NVARCHAR(10) NULL, Notes NVARCHAR(MAX) NULL, ProfileLastUpdated DATETIME2 DEFAULT GETDATE(), KarmaReceived INT NOT NULL DEFAULT 0, KarmaGiven INT NOT NULL DEFAULT 0, HateGiven INT NOT NULL DEFAULT 0);",
        "UserNameVariations": "CREATE TABLE UserNameVariations (VariationID INT IDENTITY(1,1) PRIMARY KEY, UserID BIGINT NOT NULL, NameVariation NVARCHAR(255) NOT NULL, FOREIGN KEY (UserID) REFERENCES UserProfiles(UserID) ON DELETE CASCADE);",
        "IX_UserNameVariations_UserID_NameVariation": "CREATE UNIQUE INDEX IX_UserNameVariations_UserID_NameVariation ON UserNameVariations (UserID, NameVariation);",
        "ConversationHistory": "CREATE TABLE ConversationHistory (MessageDBID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageID BIGINT NULL, Role NVARCHAR(50) NOT NULL, Content NVARCHAR(MAX) NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ConversationHistory_ChatID_Timestamp": "CREATE INDEX IX_ConversationHistory_ChatID_Timestamp ON ConversationHistory (ChatID, Timestamp DESC);",
        "ChatLog": "CREATE TABLE ChatLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, MessageID BIGINT NOT NULL, MessageText NVARCHAR(MAX) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ChatLog_ChatID_Timestamp": "CREATE INDEX IX_ChatLog_ChatID_Timestamp ON ChatLog (ChatID, Timestamp DESC);",
        "IX_ChatLog_UserID": "CREATE INDEX IX_ChatLog_UserID ON ChatLog (UserID);",
        "UserUsage": "CREATE TABLE UserUsage (UserID BIGINT NOT NULL, UsageDate DATE NOT NULL, LlmCount INT NOT NULL DEFAULT 0, ImageCount INT NOT NULL DEFAULT 0, PRIMARY KEY (UserID, UsageDate));",
        "ChatStats": "CREATE TABLE ChatStats (ChatID BIGINT PRIMARY KEY, TotalMessages INT NOT NULL DEFAULT 0, JoinCount INT NOT NULL DEFAULT 0, LeaveCount INT NOT NULL DEFAULT 0, LastUpdated DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "ChatUserStats": "CREATE TABLE ChatUserStats (ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageCount INT NOT NULL DEFAULT 0, FirstSeen DATETIME2 DEFAULT GETDATE() NOT NULL, LastActive DATETIME2 DEFAULT GETDATE() NOT NULL, PRIMARY KEY (ChatID, UserID));",
        "IX_ChatUserStats_ChatID_MessageCount": "CREATE INDEX IX_ChatUserStats_ChatID_MessageCount ON ChatUserStats (ChatID, MessageCount DESC);",
        "ChatLinkStats": "CREATE TABLE ChatLinkStats (ChatID BIGINT NOT NULL, Domain NVARCHAR(255) NOT NULL, LinkCount INT NOT NULL DEFAULT 0, PRIMARY KEY (ChatID, Domain));",
        "IX_ChatLinkStats_ChatID_Count": "CREATE INDEX IX_ChatLinkStats_ChatID_Count ON ChatLinkStats (ChatID, LinkCount DESC);",
        "ErrorLog": "CREATE TABLE ErrorLog (ErrorID INT IDENTITY(1,1) PRIMARY KEY, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, LogLevel NVARCHAR(50) NOT NULL, LoggerName NVARCHAR(255) NULL, ModuleName NVARCHAR(255) NULL, FunctionName NVARCHAR(255) NULL, LineNumber INT NULL, ErrorMessage NVARCHAR(MAX) NOT NULL, ExceptionInfo NVARCHAR(MAX) NULL);",
        "IX_ErrorLog_Timestamp": "CREATE INDEX IX_ErrorLog_Timestamp ON ErrorLog (Timestamp DESC);",
        "WebRequestLog": (
            "CREATE TABLE WebRequestLog ("
            "LogID INT IDENTITY(1,1) PRIMARY KEY,"
            "Url NVARCHAR(1024) NOT NULL,"
            "Method NVARCHAR(16) NOT NULL,"
            "StatusCode INT NULL,"
            "DurationMs INT NULL,"
            "Error NVARCHAR(512) NULL,"
            "Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL"
            ");",
        ),
        "IX_WebRequestLog_Timestamp": "CREATE INDEX IX_WebRequestLog_Timestamp ON WebRequestLog (Timestamp DESC);",
        "ChatSettings": f"CREATE TABLE ChatSettings (ChatID BIGINT PRIMARY KEY, SpamVoteThreshold INT NOT NULL DEFAULT 3, NSFWFilterEnabled BIT NOT NULL DEFAULT 0, NSFWThreshold FLOAT NOT NULL DEFAULT {config.NSFW_DETECTION_THRESHOLD});",
        "SpamReports": "CREATE TABLE SpamReports (ReportID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, TargetUserID BIGINT NOT NULL, ReporterUserID BIGINT NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, CONSTRAINT UQ_SpamReports UNIQUE (ChatID, TargetUserID, ReporterUserID));",
        "IX_SpamReports_Chat_Target": "CREATE INDEX IX_SpamReports_Chat_Target ON SpamReports (ChatID, TargetUserID);",
        "KarmaLog": "CREATE TABLE KarmaLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, GiverUserID BIGINT NOT NULL, ReceiverUserID BIGINT NOT NULL, Points INT NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_KarmaLog_ChatID_Timestamp": "CREATE INDEX IX_KarmaLog_ChatID_Timestamp ON KarmaLog (ChatID, Timestamp DESC);",
        "VerifiedUsers": "CREATE TABLE VerifiedUsers (UserID BIGINT PRIMARY KEY, VerifiedAt DATETIME2 DEFAULT GETDATE());",
        "ModerationLog": "CREATE TABLE ModerationLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NULL, MessageID BIGINT NOT NULL, Categories NVARCHAR(255) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ModerationLog_ChatID_Timestamp": "CREATE INDEX IX_ModerationLog_ChatID_Timestamp ON ModerationLog (ChatID, Timestamp DESC);",
        "FactCheckLog": "CREATE TABLE FactCheckLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, MessageID BIGINT NULL, ClaimText NVARCHAR(MAX) NOT NULL, Verdict NVARCHAR(50) NOT NULL, Confidence FLOAT NOT NULL, Track NVARCHAR(8) NULL CHECK (Track IN ('news','book')), Details NVARCHAR(MAX) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_FactCheckLog_ChatID_Timestamp": "CREATE INDEX IX_FactCheckLog_ChatID_Timestamp ON FactCheckLog (ChatID, Timestamp DESC);",
        "FactGateLog": "CREATE TABLE FactGateLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, MessageID BIGINT NOT NULL, PNews FLOAT NULL, PBook FLOAT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "assistant_invocations": (
            "CREATE TABLE assistant_invocations ("
            "id BIGINT IDENTITY PRIMARY KEY,"
            "chat_id BIGINT,"
            "user_id BIGINT,"
            "message_id BIGINT,"
            "detected BIT,"
            "alias NVARCHAR(32),"
            "prompt NVARCHAR(MAX),"
            "reason NVARCHAR(64),"
            "lang NVARCHAR(8),"
            "routed_to_llm BIT,"
            "llm_ok BIT,"
            "error NVARCHAR(512),"
            "ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()"
            ");",
        ),
        "FactBookSources": "CREATE TABLE FactBookSources (id BIGINT IDENTITY PRIMARY KEY, author NVARCHAR(256) NOT NULL, title NVARCHAR(512) NOT NULL, edition NVARCHAR(128) NULL, year INT NULL, isbn NVARCHAR(32) NULL, translator NVARCHAR(256) NULL, source_url NVARCHAR(1024) NULL, snapshot_url NVARCHAR(1024) NULL, first_seen DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME());",
        "FactBookMatches": "CREATE TABLE FactBookMatches (match_id BIGINT IDENTITY PRIMARY KEY, run_id BIGINT NOT NULL, book_source_id BIGINT NULL, quote_exact NVARCHAR(MAX) NULL, quote_lang NVARCHAR(8) NULL, page NVARCHAR(32) NULL, chapter NVARCHAR(64) NULL, stance NVARCHAR(12) NULL, score FLOAT NULL);",
        "IX_FactBookMatches_Run": "CREATE INDEX IX_FactBookMatches_Run ON FactBookMatches(run_id);",
        # ------------------------------------------------------------------
        # Memory/portrait feature tables
        # ------------------------------------------------------------------
        "answer_evidence": (
            "CREATE TABLE answer_evidence ("
            "answer_id BIGINT IDENTITY PRIMARY KEY,"
            "chat_id BIGINT NOT NULL,"
            "asked_by BIGINT NOT NULL,"
            "intent NVARCHAR(32) NOT NULL,"
            "query_text NVARCHAR(MAX) NULL,"
            "lang NVARCHAR(8) NULL,"
            "created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()"
            ");"
        ),
        "answer_evidence_items": (
            "CREATE TABLE answer_evidence_items ("
            "answer_id BIGINT NOT NULL,"
            "message_id BIGINT NOT NULL,"
            "rank INT NOT NULL,"
            "snippet NVARCHAR(MAX) NULL,"
            "reason NVARCHAR(64) NULL,"
            "PRIMARY KEY(answer_id, message_id)"
            ");"
        ),
        "user_persona_versions": (
            "CREATE TABLE user_persona_versions ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "version INT NOT NULL,"
            "created_at DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),"
            "portrait_md NVARCHAR(MAX) NOT NULL,"
            "traits_json NVARCHAR(MAX) NOT NULL,"
            "signals_json NVARCHAR(MAX) NULL,"
            "PRIMARY KEY(chat_id, user_id, version)"
            ");"
        ),
        "IX_answer_evidence_chat": "CREATE INDEX IX_answer_evidence_chat ON answer_evidence(chat_id, created_at DESC);",
        "IX_user_persona_versions_lookup": "CREATE INDEX IX_user_persona_versions_lookup ON user_persona_versions(chat_id, user_id, version DESC);",
        # ------------------------------------------------------------------
        # Advanced karma system tables
        # ------------------------------------------------------------------
        "karma_events": (
            "CREATE TABLE karma_events ("
            "event_id BIGINT IDENTITY(1,1) PRIMARY KEY,"
            "chat_id BIGINT NOT NULL,"
            "msg_id BIGINT NULL,"
            "target_user_id BIGINT NOT NULL,"
            "rater_user_id BIGINT NOT NULL,"
            "emoji NVARCHAR(16) NULL,"
            "base FLOAT NOT NULL,"
            "rater_trust FLOAT NOT NULL,"
            "diversity FLOAT NOT NULL,"
            "anti_collusion FLOAT NOT NULL,"
            "novelty FLOAT NOT NULL,"
            "content_factor FLOAT NOT NULL,"
            "weight FLOAT NOT NULL,"
            "ts DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()"
            ");"
        ),
        "IX_events_chat_msg": "CREATE INDEX IX_events_chat_msg ON karma_events(chat_id, msg_id) INCLUDE (ts, weight);",
        "message_scores": (
            "CREATE TABLE message_scores ("
            "chat_id BIGINT NOT NULL,"
            "msg_id BIGINT NOT NULL,"
            "author_id BIGINT NOT NULL,"
            "score_current FLOAT NOT NULL DEFAULT 0,"
            "pos INT NOT NULL DEFAULT 0,"
            "neg INT NOT NULL DEFAULT 0,"
            "last_update DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),"
            "CONSTRAINT PK_message_scores PRIMARY KEY (chat_id, msg_id)"
            ");"
        ),
        "IX_msg_author": "CREATE INDEX IX_msg_author ON message_scores(author_id, last_update DESC);",
        "user_rep_rollup": (
            "CREATE TABLE user_rep_rollup ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "day DATE NOT NULL,"
            "delta_score FLOAT NOT NULL,"
            "pos INT NOT NULL,"
            "neg INT NOT NULL,"
            "skills_json NVARCHAR(MAX) NULL,"
            "CONSTRAINT PK_user_rep_rollup PRIMARY KEY (chat_id, user_id, day)"
            ");"
        ),
        "IX_user_rep_rollup_day": "CREATE INDEX IX_user_rep_rollup_day ON user_rep_rollup(day);",
        "user_rep_current": (
            "CREATE TABLE user_rep_current ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "rep FLOAT NOT NULL DEFAULT 0,"
            "rep_global FLOAT NOT NULL DEFAULT 0,"
            "streak_days INT NOT NULL DEFAULT 0,"
            "last_seen DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),"
            "badges_json NVARCHAR(MAX) NULL,"
            "CONSTRAINT PK_user_rep_current PRIMARY KEY (chat_id, user_id)"
            ");"
        ),
        "IX_user_rep_rank": "CREATE INDEX IX_user_rep_rank ON user_rep_current(chat_id, rep DESC);",
        "skill_rep_current": (
            "CREATE TABLE skill_rep_current ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "tag NVARCHAR(64) NOT NULL,"
            "rep FLOAT NOT NULL,"
            "CONSTRAINT PK_skill_rep_current PRIMARY KEY (chat_id, user_id, tag)"
            ");"
        ),
        "IX_skill_tag": "CREATE INDEX IX_skill_tag ON skill_rep_current(tag, rep DESC);",
        "karmaconfig": (
            "CREATE TABLE karmaconfig ("
            "chat_id BIGINT PRIMARY KEY,"
            "emoji_map_json NVARCHAR(MAX) NULL,"
            "decay_msg_days INT NOT NULL DEFAULT 7,"
            "decay_user_days INT NOT NULL DEFAULT 45,"
            "allow_downvotes BIT NOT NULL DEFAULT 1,"
            "daily_budget INT NOT NULL DEFAULT 18,"
            "downvote_quorum INT NOT NULL DEFAULT 4,"
            "diversity_window_hours INT NOT NULL DEFAULT 12,"
            "reciprocity_threshold FLOAT NOT NULL DEFAULT 0.30,"
            "preset NVARCHAR(32) NULL,"
            "auto_tune BIT NOT NULL DEFAULT 1"
            ");"
        ),
        "trust_table": (
            "CREATE TABLE trust_table ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "trust FLOAT NOT NULL,"
            "upheld INT NOT NULL DEFAULT 0,"
            "overturned INT NOT NULL DEFAULT 0,"
            "tenure_days INT NOT NULL DEFAULT 0,"
            "phone_verified BIT NOT NULL DEFAULT 0,"
            "last_update DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),"
            "CONSTRAINT PK_trust_table PRIMARY KEY (chat_id, user_id)"
            ");"
        ),
        "karmabans": (
            "CREATE TABLE karmabans ("
            "chat_id BIGINT NOT NULL,"
            "user_id BIGINT NOT NULL,"
            "reason NVARCHAR(256) NULL,"
            "until_ts DATETIME2 NULL,"
            "CONSTRAINT PK_karmabans PRIMARY KEY (chat_id, user_id)"
            ");"
        ),
    }
    try:
        with conn.cursor() as cursor:
            logger.info("Initializing database tables...")
            for name, query in table_queries.items():
                # Some queries are defined as tuples for readability; join them into a single string.
                query_str = (
                    " ".join(query) if isinstance(query, (tuple, list)) else query
                )

                is_idx = name.startswith("IX_")
                obj_type = "INDEX" if is_idx else "TABLE"
                obj_name_to_check = name  # For tables, this is the table name. For indexes, this is the index name.
                table_for_index = ""
                if is_idx:
                    # Extract table name from the CREATE INDEX statement.
                    # Use a word boundary before 'ON' so we don't match strings like
                    # 'NameVariation ON'.
                    match = re.search(r"\bON\s+([\w\.]+)", query_str, re.IGNORECASE)
                    if match:
                        table_for_index = match.group(1)
                    else:
                        logger.warning(f"Could not determine table for index {name}")
                        continue

                check_q = (
                    "SELECT 1 FROM sys.indexes i INNER JOIN sys.objects o ON i.object_id = o.object_id "
                    "WHERE i.name = ? AND o.name = ?"
                    if is_idx
                    else "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
                )
                params_check = (
                    (obj_name_to_check, table_for_index)
                    if is_idx
                    else (obj_name_to_check,)
                )

                cursor.execute(check_q, params_check)
                if cursor.fetchone():
                    logger.info(f"{obj_type} '{obj_name_to_check}' already exists.")
                else:
                    logger.info(
                        f"{obj_type} '{obj_name_to_check}' not found. Creating..."
                    )
                    cursor.execute(query_str)
                    logger.info(f"{obj_type} '{obj_name_to_check}' created.")

            # Ensure karma-related columns exist on UserProfiles for backwards compatibility
            karma_columns = {
                "KarmaReceived": "INT NOT NULL DEFAULT 0",
                "KarmaGiven": "INT NOT NULL DEFAULT 0",
                "HateGiven": "INT NOT NULL DEFAULT 0",
            }
            for col_name, definition in karma_columns.items():
                cursor.execute(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'UserProfiles' AND COLUMN_NAME = ?",
                    (col_name,),
                )
                if cursor.fetchone():
                    logger.info(
                        f"Column '{col_name}' already exists in 'UserProfiles'."
                    )
                else:
                    logger.info(
                        f"Column '{col_name}' missing in 'UserProfiles'. Adding..."
                    )
                    cursor.execute(
                        f"ALTER TABLE UserProfiles ADD {col_name} {definition}"
                    )
                    logger.info(f"Column '{col_name}' added to 'UserProfiles'.")

            # Ensure fact-check log has optional columns
            fc_columns = {
                "Track": "NVARCHAR(8) NULL CHECK (Track IN ('news','book'))",
                "Details": "NVARCHAR(MAX) NULL",
            }
            for col_name, definition in fc_columns.items():
                cursor.execute(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'FactCheckLog' AND COLUMN_NAME = ?",
                    (col_name,),
                )
                if cursor.fetchone():
                    logger.info(
                        f"Column '{col_name}' already exists in 'FactCheckLog'."
                    )
                else:
                    logger.info(
                        f"Column '{col_name}' missing in 'FactCheckLog'. Adding..."
                    )
                    cursor.execute(
                        f"ALTER TABLE FactCheckLog ADD {col_name} {definition}"
                    )
                    logger.info(
                        f"Column '{col_name}' added to 'FactCheckLog'."
                    )

            logger.info("Database initialization check complete.")
    except Exception as e:
        logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn:
            conn.autocommit = False
            conn.close()
