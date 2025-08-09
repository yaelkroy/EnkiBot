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
import pyodbc
from typing import List, Dict, Any, Optional

from enkibot import config # For DB_CONNECTION_STRING

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.connection_string = connection_string
        if not self.connection_string:
            logger.warning("Database connection string is not configured. Database operations will be disabled.")

    def get_db_connection(self) -> Optional[pyodbc.Connection]:
        if not self.connection_string: return None
        try:
            conn = pyodbc.connect(self.connection_string, autocommit=False)
            logger.debug("Database connection established.")
            return conn
        except pyodbc.Error as ex:
            logger.error(f"Database connection error: {ex.args[0] if ex.args else ''} - {ex}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error establishing database connection: {e}", exc_info=True)
            return None

    async def execute_query(self, query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = False):
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
            logger.error(f"DB query error on '{query[:100]}...': {ex}", exc_info=True)
            try: conn.rollback(); logger.info("Transaction rolled back.")
            except pyodbc.Error as rb_ex: logger.error(f"Error during rollback: {rb_ex}", exc_info=True)
            return None if fetch_one or fetch_all else False
        except Exception as e:
            logger.error(f"Unexpected error query execution '{query[:100]}...': {e}", exc_info=True)
            return None if fetch_one or fetch_all else False
        finally:
             if conn:
                conn.close()
                logger.debug("DB connection closed post-exec.")

    async def get_recent_chat_texts(self, chat_id: int, limit: int = 3) -> List[str]:
        if not self.connection_string: return []
        query = """
            SELECT TOP (?) MessageText FROM ChatLog
            WHERE ChatID = ? AND MessageText IS NOT NULL AND RTRIM(LTRIM(MessageText)) != ''
            ORDER BY Timestamp DESC
        """
        actual_limit = max(1, limit) 
        rows = await self.execute_query(query, (actual_limit, chat_id), fetch_all=True)
        return [row.MessageText for row in reversed(rows) if row.MessageText] if rows else []

    async def log_chat_message_and_upsert_user(
        self, chat_id: int, user_id: int, username: Optional[str],
        first_name: Optional[str], last_name: Optional[str],
        message_id: int, message_text: str, preferred_language: Optional[str] = None
    ) -> Optional[str]:
        if not self.connection_string: return None
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
        chat_log_params = (chat_id, user_id, username, first_name, message_id, message_text)
        conn = self.get_db_connection()
        if not conn: return None
        action_taken = None
        try:
            with conn.cursor() as cursor:
                cursor.execute(upsert_user_sql, user_params)
                row = cursor.fetchone();
                if row: action_taken = row.Action
                cursor.execute(chat_log_sql, chat_log_params)
            conn.commit()
            logger.info(f"Logged message for user {user_id}. Profile action: {action_taken}")
            return action_taken
        except pyodbc.Error as ex:
            logger.error(f"DB error logging/upserting user {user_id}: {ex}", exc_info=True); conn.rollback()
        finally:
            if conn: conn.close()
        return None # Ensure a return path if try fails before commit

    async def get_conversation_history(self, chat_id: int, limit: int = 20) -> List[Dict[str, str]]:
        query = "SELECT TOP (?) Role, Content FROM ConversationHistory WHERE ChatID = ? ORDER BY Timestamp DESC"
        rows = await self.execute_query(query, (limit, chat_id), fetch_all=True)
        return [{"role": row.Role.lower(), "content": row.Content} for row in reversed(rows)] if rows else []

    async def save_to_conversation_history(self, chat_id: int, entity_id: int, message_id_telegram: Optional[int], role: str, content: str):
        query = "INSERT INTO ConversationHistory (ChatID, UserID, MessageID, Role, Content, Timestamp) VALUES (?, ?, ?, ?, ?, GETDATE())"
        await self.execute_query(query, (chat_id, entity_id, message_id_telegram, role, content), commit=True)

    async def get_user_profile_notes(self, user_id: int) -> Optional[str]:
        row = await self.execute_query("SELECT Notes FROM UserProfiles WHERE UserID = ?", (user_id,), fetch_one=True)
        return row.Notes if row and row.Notes else None

    async def update_user_profile_notes(self, user_id: int, notes: str):
        await self.execute_query("UPDATE UserProfiles SET Notes = ?, ProfileLastUpdated = GETDATE() WHERE UserID = ?", (notes, user_id), commit=True)
        logger.info(f"Profile notes updated for user {user_id}.")

    async def save_user_name_variations(self, user_id: int, variations: List[str]):
        if not self.connection_string or not variations: return
        sql_merge = """
            MERGE INTO UserNameVariations AS t USING (SELECT ? AS UserID, ? AS NameVariation) AS s
            ON (t.UserID = s.UserID AND t.NameVariation = s.NameVariation)
            WHEN NOT MATCHED THEN INSERT (UserID, NameVariation) VALUES (s.UserID, s.NameVariation);
        """
        conn = self.get_db_connection()
        if not conn: return
        try:
            with conn.cursor() as cursor:
                params_to_insert = [(user_id, var) for var in variations if var and str(var).strip()]
                if params_to_insert: cursor.executemany(sql_merge, params_to_insert)
            conn.commit()
            logger.info(f"Saved/updated {len(params_to_insert)} name vars for user {user_id}.")
        except pyodbc.Error as ex:
            logger.error(f"DB error saving name vars for user {user_id}: {ex}", exc_info=True); conn.rollback()
        finally:
            if conn: conn.close()
            
    async def find_user_profiles_by_name_variation(self, name_variation_query: str) -> List[Dict[str, Any]]:
        query = """
            SELECT DISTINCT up.UserID, up.FirstName, up.LastName, up.Username, up.Notes
            FROM UserProfiles up JOIN UserNameVariations unv ON up.UserID = unv.UserID
            WHERE unv.NameVariation = ?
        """
        rows = await self.execute_query(query, (name_variation_query.lower(),), fetch_all=True)
        return [{"UserID": r.UserID, "FirstName": r.FirstName, "LastName": r.LastName, "Username": r.Username, "Notes": r.Notes} for r in rows] if rows else []

    async def get_user_messages_from_chat_log(self, user_id: int, chat_id: int, limit: int = 10) -> List[str]: # Kept from previous version
        query = "SELECT TOP (?) MessageText FROM ChatLog WHERE UserID = ? AND ChatID = ? AND MessageText IS NOT NULL AND RTRIM(LTRIM(MessageText)) != '' ORDER BY Timestamp DESC"
        rows = await self.execute_query(query, (limit, user_id, chat_id), fetch_all=True)
        return [row.MessageText for row in rows] if rows else []

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

    async def add_spam_vote(self, chat_id: int, target_user_id: int, reporter_user_id: int) -> bool:
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
            return bool(row and row.Action == 'INSERT')
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

    async def count_spam_votes(self, chat_id: int, target_user_id: int, window_minutes: int) -> int:
        """Returns the number of unique reporters within the given timeframe."""
        query = (
            "SELECT COUNT(DISTINCT ReporterUserID) AS VoteCount FROM SpamReports "
            "WHERE ChatID = ? AND TargetUserID = ? AND Timestamp > DATEADD(MINUTE, ?, GETDATE())"
        )
        row = await self.execute_query(query, (chat_id, target_user_id, -abs(window_minutes)), fetch_one=True)
        return row.VoteCount if row and hasattr(row, 'VoteCount') else 0

    async def clear_spam_votes(self, chat_id: int, target_user_id: int):
        await self.execute_query(
            "DELETE FROM SpamReports WHERE ChatID = ? AND TargetUserID = ?",
            (chat_id, target_user_id),
            commit=True,
        )

    async def get_spam_vote_threshold(self, chat_id: int, default_threshold: int) -> int:
        row = await self.execute_query(
            "SELECT SpamVoteThreshold FROM ChatSettings WHERE ChatID = ?",
            (chat_id,),
            fetch_one=True,
        )
        if row and hasattr(row, 'SpamVoteThreshold') and row.SpamVoteThreshold:
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

    async def add_warning(self, chat_id: int, user_id: int, reason: Optional[str] = None) -> int:
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
            return row.WarnCount if row and hasattr(row, 'WarnCount') else 0
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
        return row.WarnCount if row and hasattr(row, 'WarnCount') else 0

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

def initialize_database(): # This function defines and uses DatabaseManager locally
    if not config.DB_CONNECTION_STRING:
        logger.warning("Cannot initialize database: Connection string not configured.")
        return
    
    db_mngr = DatabaseManager(config.DB_CONNECTION_STRING) 
    conn = db_mngr.get_db_connection() 
    if not conn: 
        logger.error("Failed to connect to database for initialization (initialize_database).")
        return
    
    conn.autocommit = True 
    table_queries = {
        "UserProfiles": "CREATE TABLE UserProfiles (UserID BIGINT PRIMARY KEY, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, LastName NVARCHAR(255) NULL, LastSeen DATETIME2 DEFAULT GETDATE(), MessageCount INT DEFAULT 0, PreferredLanguage NVARCHAR(10) NULL, Notes NVARCHAR(MAX) NULL, ProfileLastUpdated DATETIME2 DEFAULT GETDATE());",
        "UserNameVariations": "CREATE TABLE UserNameVariations (VariationID INT IDENTITY(1,1) PRIMARY KEY, UserID BIGINT NOT NULL, NameVariation NVARCHAR(255) NOT NULL, FOREIGN KEY (UserID) REFERENCES UserProfiles(UserID) ON DELETE CASCADE);",
        "IX_UserNameVariations_UserID_NameVariation": "CREATE UNIQUE INDEX IX_UserNameVariations_UserID_NameVariation ON UserNameVariations (UserID, NameVariation);",
        "ConversationHistory": "CREATE TABLE ConversationHistory (MessageDBID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageID BIGINT NULL, Role NVARCHAR(50) NOT NULL, Content NVARCHAR(MAX) NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ConversationHistory_ChatID_Timestamp": "CREATE INDEX IX_ConversationHistory_ChatID_Timestamp ON ConversationHistory (ChatID, Timestamp DESC);",
        "ChatLog": "CREATE TABLE ChatLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, MessageID BIGINT NOT NULL, MessageText NVARCHAR(MAX) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ChatLog_ChatID_Timestamp": "CREATE INDEX IX_ChatLog_ChatID_Timestamp ON ChatLog (ChatID, Timestamp DESC);",
        "IX_ChatLog_UserID": "CREATE INDEX IX_ChatLog_UserID ON ChatLog (UserID);",
        "ChatStats": "CREATE TABLE ChatStats (ChatID BIGINT PRIMARY KEY, TotalMessages INT NOT NULL DEFAULT 0, JoinCount INT NOT NULL DEFAULT 0, LeaveCount INT NOT NULL DEFAULT 0, LastUpdated DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "ChatUserStats": "CREATE TABLE ChatUserStats (ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageCount INT NOT NULL DEFAULT 0, FirstSeen DATETIME2 DEFAULT GETDATE() NOT NULL, LastActive DATETIME2 DEFAULT GETDATE() NOT NULL, PRIMARY KEY (ChatID, UserID));",
        "IX_ChatUserStats_ChatID_MessageCount": "CREATE INDEX IX_ChatUserStats_ChatID_MessageCount ON ChatUserStats (ChatID, MessageCount DESC);",
        "ChatLinkStats": "CREATE TABLE ChatLinkStats (ChatID BIGINT NOT NULL, Domain NVARCHAR(255) NOT NULL, LinkCount INT NOT NULL DEFAULT 0, PRIMARY KEY (ChatID, Domain));",
        "IX_ChatLinkStats_ChatID_Count": "CREATE INDEX IX_ChatLinkStats_ChatID_Count ON ChatLinkStats (ChatID, LinkCount DESC);",
        "ErrorLog": "CREATE TABLE ErrorLog (ErrorID INT IDENTITY(1,1) PRIMARY KEY, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, LogLevel NVARCHAR(50) NOT NULL, LoggerName NVARCHAR(255) NULL, ModuleName NVARCHAR(255) NULL, FunctionName NVARCHAR(255) NULL, LineNumber INT NULL, ErrorMessage NVARCHAR(MAX) NOT NULL, ExceptionInfo NVARCHAR(MAX) NULL);",
        "IX_ErrorLog_Timestamp": "CREATE INDEX IX_ErrorLog_Timestamp ON ErrorLog (Timestamp DESC);",
        "ChatSettings": "CREATE TABLE ChatSettings (ChatID BIGINT PRIMARY KEY, SpamVoteThreshold INT NOT NULL DEFAULT 3);",
        "SpamReports": "CREATE TABLE SpamReports (ReportID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, TargetUserID BIGINT NOT NULL, ReporterUserID BIGINT NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, CONSTRAINT UQ_SpamReports UNIQUE (ChatID, TargetUserID, ReporterUserID));",
        "IX_SpamReports_Chat_Target": "CREATE INDEX IX_SpamReports_Chat_Target ON SpamReports (ChatID, TargetUserID);",
        "VerifiedUsers": "CREATE TABLE VerifiedUsers (UserID BIGINT PRIMARY KEY, VerifiedAt DATETIME2 DEFAULT GETDATE());",
    }
    try:
        with conn.cursor() as cursor:
            logger.info("Initializing database tables...")
            for name, query in table_queries.items():
                is_idx = name.startswith("IX_")
                obj_type = "INDEX" if is_idx else "TABLE"
                obj_name_to_check = name # For tables, this is the table name. For indexes, this is the index name.
                table_for_index = ""
                if is_idx:
                    # Attempt to parse table name from index name, e.g., IX_TableName_Column -> TableName
                    parts = name.split('_')
                    if len(parts) > 1: table_for_index = parts[1] # This is a heuristic
                    else: logger.warning(f"Could not determine table for index {name}"); continue 
                
                check_q = "SELECT name FROM sys.indexes WHERE name = ? AND object_id = OBJECT_ID(?)" if is_idx else "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
                params_check = (obj_name_to_check, table_for_index) if is_idx else (obj_name_to_check,)
                
                cursor.execute(check_q, params_check)
                if cursor.fetchone(): logger.info(f"{obj_type} '{obj_name_to_check}' already exists.")
                else: logger.info(f"{obj_type} '{obj_name_to_check}' not found. Creating..."); cursor.execute(query); logger.info(f"{obj_type} '{obj_name_to_check}' created.")
            logger.info("Database initialization check complete.")
    except Exception as e:
        logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn:
            conn.autocommit = False
            conn.close()
