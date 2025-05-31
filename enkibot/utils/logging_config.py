# enkibot/utils/logging_config.py
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
# === EnkiBot Logging Configuration ===
# ==================================================================================================
# Sets up application-wide structured logging to both file and console.
# Includes a custom logging handler to write critical errors to the SQL database
# and clears the previous log file on startup.
# ==================================================================================================

import logging
import traceback
import pyodbc
import os # <--- IMPORT OS MODULE
from enkibot import config

def setup_logging():
    """Initializes the logging configuration for the entire application."""

    # --- START: CLEAR PREVIOUS LOG FILE ---
    log_file_name = "bot_activity.log"
    try:
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
            # This print goes to console before logging is fully set up for the bot's logger.
            print(f"INFO: Previous log file '{log_file_name}' removed successfully.")
    except OSError as e:
        # This print also goes to console.
        print(f"WARNING: Error removing previous log file '{log_file_name}': {e}")
    # --- END: CLEAR PREVIOUS LOG FILE ---
    
    # Define a custom handler for logging errors to the database
    class SQLDBLogHandler(logging.Handler):
        """
        A logging handler that writes log records with level ERROR or higher
        to a dedicated table in the SQL Server database.
        """
        def __init__(self):
            super().__init__()
            self.conn = None

        def _get_db_conn_for_logging(self):
            """Establishes a database connection specifically for logging."""
            if not config.DB_CONNECTION_STRING:
                return None
            try:
                # Use autocommit=True for logging to ensure errors are written immediately.
                return pyodbc.connect(config.DB_CONNECTION_STRING, autocommit=True)
            except pyodbc.Error:
                # If the DB is down, we can't log to it. Silently fail for now.
                # A print statement could be added here for immediate feedback if needed.
                # print("WARNING: SQLDBLogHandler could not connect to the database for logging.")
                return None

        def emit(self, record: logging.LogRecord):
            """
            Writes the log record to the ErrorLog table in the database.
            """
            if self.conn is None:
                self.conn = self._get_db_conn_for_logging()

            if self.conn:
                try:
                    msg = self.format(record)
                    exc_info_str = traceback.format_exc() if record.exc_info else None
                    sql = "INSERT INTO ErrorLog (LogLevel, LoggerName, ModuleName, FunctionName, LineNumber, ErrorMessage, ExceptionInfo) VALUES (?, ?, ?, ?, ?, ?, ?)"
                    with self.conn.cursor() as cursor:
                        cursor.execute(sql, record.levelname, record.name, record.module, record.funcName, record.lineno, msg, exc_info_str)
                except pyodbc.Error:
                    # If an error occurs during logging, handle it and sever the connection.
                    self.handleError(record)
                    self.conn = None # Reset connection to be re-established on next emit.
        
        def close(self):
            """Closes the database connection if it's open."""
            if self.conn:
                try:
                    self.conn.close()
                except pyodbc.Error:
                    pass
            super().close()

    # --- Main Logging Configuration ---
    log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    
    # Configure root logger
    # By using basicConfig with force=True (Python 3.8+), we can reconfigure if needed.
    # For older Python, ensure basicConfig is called only once application-wide.
    # Given setup_logging() is called once from main.py, this should be fine.
    logging.basicConfig(
        format=log_format,
        level=log_level,
        handlers=[
            logging.FileHandler(log_file_name, encoding='utf-8'), # Use variable
            logging.StreamHandler()
        ]
        # force=True # Add this if using Python 3.8+ and re-running setup_logging,
                   # but it should not be necessary with current structure.
    )
    
    module_logger = logging.getLogger(__name__) # Logger for this specific module (logging_config.py)

    # Add the custom DB handler if the database is configured
    if config.DB_CONNECTION_STRING:
        db_log_handler = SQLDBLogHandler()
        db_log_handler.setLevel(logging.ERROR) # Only log ERROR and CRITICAL to DB
        formatter = logging.Formatter(log_format)
        db_log_handler.setFormatter(formatter)
        logging.getLogger().addHandler(db_log_handler) # Add to the root logger
        module_logger.info("Configured logging of ERROR-level messages to the SQL database.")
    else:
        module_logger.warning("Logging to SQL database is NOT configured (DB_CONNECTION_STRING is missing).")