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
# ==================================================================================================
# === EnkiBot Database Utilities ===
# ==================================================================================================
# Handles all database interactions for EnkiBot, including setup,
# connection management, and data querying/storage.
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
        """Establishes and returns a new database connection."""
        if not self.connection_string:
            return None
        try:
            # autocommit=False by default, manage transactions explicitly
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
        """Executes a given SQL query with optional parameters and commit."""
        if not self.connection_string:
            logger.warning("Query execution skipped: Database not configured.")
            return None if fetch_one or fetch_all else (True if commit else False)


        conn = self.get_db_connection()
        if not conn:
            return None if fetch_one or fetch_all else (True if commit else False)

        try:
            with conn.cursor() as cursor:
                logger.debug(f"Executing query: {query[:150]}... with params: {params}")
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if commit:
                    conn.commit()
                    logger.debug("Query committed successfully.")
                    return True

                if fetch_one:
                    row = cursor.fetchone()
                    logger.debug(f"Query fetch_one result: {row}")
                    return row
                if fetch_all:
                    rows = cursor.fetchall()
                    logger.debug(f"Query fetch_all result count: {len(rows)}")
                    return rows
            return True # Default success for non-fetching, non-committing queries
        except pyodbc.Error as ex:
            logger.error(f"Database query error on '{query[:100]}...': {ex}", exc_info=True)
            try: conn.rollback(); logger.info("Transaction rolled back.")
            except pyodbc.Error as rb_ex: logger.error(f"Error during rollback: {rb_ex}", exc_info=True)
            return None if fetch_one or fetch_all else False
        except Exception as e:
            logger.error(f"Unexpected error during query execution '{query[:100]}...': {e}", exc_info=True)
            return None if fetch_one or fetch_all else False
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed post-execution.")
    async def get_recent_chat_texts(self, chat_id: int, limit: int = 3) -> List[str]:
        """
        Fetches the text of the last N messages from a given chat_id from ChatLog.
        Filters out potential null or empty messages.
        """
        if not self.connection_string:
            return []
        
        query = """
            SELECT TOP (?) MessageText
            FROM ChatLog
            WHERE ChatID = ? AND MessageText IS NOT NULL AND MessageText != ''
            ORDER BY Timestamp DESC
        """
        # Ensure 'limit' is a positive integer
        actual_limit = max(1, limit) 
        
        rows = await self.execute_query(query, (actual_limit, chat_id), fetch_all=True)
        if rows:
            # Messages are fetched in reverse chronological order, reverse again for natural order
            return [row.MessageText for row in reversed(rows) if row.MessageText] 
        return []
    async def log_chat_message_and_upsert_user(
        self, chat_id: int, user_id: int, username: Optional[str],
        first_name: Optional[str], last_name: Optional[str],
        message_id: int, message_text: str, preferred_language: Optional[str] = None
    ) -> Optional[str]:
        """Logs a chat message and creates or updates the user profile."""
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
                row = cursor.fetchone()
                if row: action_taken = row.Action
                cursor.execute(chat_log_sql, chat_log_params)
            conn.commit()
            logger.info(f"Logged message for user {user_id}. Profile action: {action_taken}")
            return action_taken
        except pyodbc.Error as ex:
            logger.error(f"DB error logging/upserting user {user_id}: {ex}", exc_info=True); conn.rollback()
            return None
        finally:
            if conn: conn.close()

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

    async def get_user_messages_from_chat_log(self, user_id: int, chat_id: int, limit: int = 10) -> List[str]:
        query = "SELECT TOP (?) MessageText FROM ChatLog WHERE UserID = ? AND ChatID = ? AND MessageText IS NOT NULL AND MessageText != '' ORDER BY Timestamp DESC"
        rows = await self.execute_query(query, (limit, user_id, chat_id), fetch_all=True)
        return [row.MessageText for row in rows] if rows else []

def initialize_database():
    if not config.DB_CONNECTION_STRING:
        logger.warning("Cannot initialize database: Connection string not configured.")
        return
    db_mngr = DatabaseManager(config.DB_CONNECTION_STRING) # Use the manager
    conn = db_mngr.get_db_connection() # Get connection via manager
    if not conn: return
    conn.autocommit = True 
    table_queries = { # Simplified from your example for brevity, ensure all are there
        "UserProfiles": "CREATE TABLE UserProfiles (UserID BIGINT PRIMARY KEY, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, LastName NVARCHAR(255) NULL, LastSeen DATETIME2 DEFAULT GETDATE(), MessageCount INT DEFAULT 0, PreferredLanguage NVARCHAR(10) NULL, Notes NVARCHAR(MAX) NULL, ProfileLastUpdated DATETIME2 DEFAULT GETDATE());",
        "UserNameVariations": "CREATE TABLE UserNameVariations (VariationID INT IDENTITY(1,1) PRIMARY KEY, UserID BIGINT NOT NULL, NameVariation NVARCHAR(255) NOT NULL, FOREIGN KEY (UserID) REFERENCES UserProfiles(UserID) ON DELETE CASCADE);",
        "IX_UserNameVariations_UserID_NameVariation": "CREATE UNIQUE INDEX IX_UserNameVariations_UserID_NameVariation ON UserNameVariations (UserID, NameVariation);",
        "ConversationHistory": "CREATE TABLE ConversationHistory (MessageDBID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, MessageID BIGINT NULL, Role NVARCHAR(50) NOT NULL, Content NVARCHAR(MAX) NOT NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ConversationHistory_ChatID_Timestamp": "CREATE INDEX IX_ConversationHistory_ChatID_Timestamp ON ConversationHistory (ChatID, Timestamp DESC);",
        "ChatLog": "CREATE TABLE ChatLog (LogID INT IDENTITY(1,1) PRIMARY KEY, ChatID BIGINT NOT NULL, UserID BIGINT NOT NULL, Username NVARCHAR(255) NULL, FirstName NVARCHAR(255) NULL, MessageID BIGINT NOT NULL, MessageText NVARCHAR(MAX) NULL, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL);",
        "IX_ChatLog_ChatID_Timestamp": "CREATE INDEX IX_ChatLog_ChatID_Timestamp ON ChatLog (ChatID, Timestamp DESC);",
        "IX_ChatLog_UserID": "CREATE INDEX IX_ChatLog_UserID ON ChatLog (UserID);",
        "ErrorLog": "CREATE TABLE ErrorLog (ErrorID INT IDENTITY(1,1) PRIMARY KEY, Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL, LogLevel NVARCHAR(50) NOT NULL, LoggerName NVARCHAR(255) NULL, ModuleName NVARCHAR(255) NULL, FunctionName NVARCHAR(255) NULL, LineNumber INT NULL, ErrorMessage NVARCHAR(MAX) NOT NULL, ExceptionInfo NVARCHAR(MAX) NULL);",
        "IX_ErrorLog_Timestamp": "CREATE INDEX IX_ErrorLog_Timestamp ON ErrorLog (Timestamp DESC);",
    }
    try:
        with conn.cursor() as cursor:
            logger.info("Initializing database tables...")
            for name, query in table_queries.items():
                is_idx = name.startswith("IX_")
                obj_type = "INDEX" if is_idx else "TABLE"
                obj_name = name if is_idx else name
                tbl_for_idx = name.split("_on_")[-1] if "_on_" in name else (name.split("_")[1] if is_idx and len(name.split("_")) > 1 else name)
                
                check_q = "SELECT name FROM sys.indexes WHERE name = ? AND object_id = OBJECT_ID(?)" if is_idx else "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?"
                params_check = (obj_name, tbl_for_idx) if is_idx else (obj_name,)
                
                cursor.execute(check_q, params_check)
                if cursor.fetchone(): logger.info(f"{obj_type} '{obj_name}' already exists.")
                else: logger.info(f"{obj_type} '{obj_name}' not found. Creating..."); cursor.execute(query); logger.info(f"{obj_type} '{obj_name}' created.")
            logger.info("Database initialization check complete.")
    except Exception as e: logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn: conn.autocommit = False; conn.close()

# backfill_existing_user_name_variations should be moved to main.py or a utility script
# to be run in a context where ProfileManager is available.