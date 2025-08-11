# enkibot/main.py
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
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
import os
import logging
from typing import Optional

try:
    from telegram import Update
    from telegram.ext import Application
    from telegram.request import HTTPXRequest
except ImportError as exc:
    raise ImportError(
        "EnkiBot requires the 'python-telegram-bot' package. Install it with "
        "`pip install python-telegram-bot>=20.0` and make sure any conflicting "
        "`telegram` package is uninstalled."
    ) from exc

from enkibot import config
from enkibot.utils.logging_config import setup_logging
from enkibot.utils.database import initialize_database
# --- MODIFIED IMPORT ---
from enkibot.app import EnkiBotApplication 
# --- END MODIFICATION ---

logger: Optional[logging.Logger] = None 

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# Commented out backfill for now, as it needs careful setup if run from here
# async def run_backfill_async(): ...

def main() -> None:
    global logger 
    clear_terminal()
    setup_logging() 
    logger = logging.getLogger(__name__) 

    logger.info("Initializing database schema...")
    initialize_database()

    if not config.TELEGRAM_BOT_TOKEN:
        logger.critical("FATAL: TELEGRAM_BOT_TOKEN missing. Bot cannot start.")
        return

    try:
        logger.info("Initializing Telegram PTB Application...")
        request = HTTPXRequest(
            connect_timeout=config.TELEGRAM_CONNECT_TIMEOUT,
            read_timeout=config.TELEGRAM_READ_TIMEOUT,
            write_timeout=config.TELEGRAM_WRITE_TIMEOUT,
            pool_timeout=config.TELEGRAM_POOL_TIMEOUT,
        )

        # Placeholder for the bot application instance so it can be referenced
        # inside the post-init callback.
        enkibot_app_instance: EnkiBotApplication | None = None

        async def post_init(application: Application) -> None:
            """Publish default commands once the application is running."""
            if enkibot_app_instance:
                await enkibot_app_instance.handler_service.push_default_commands()

        ptb_app = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .request(request)
            .build()
        )
        # Register the post_init callback without calling it (avoids NoneType errors)
        ptb_app.post_init = post_init

        logger.info("Initializing EnkiBotApplication...")
        # --- MODIFIED BOT INSTANTIATION ---
        enkibot_app_instance = EnkiBotApplication(ptb_app)
        enkibot_app_instance.register_handlers()  # Call the method to register handlers
        # --- END MODIFICATION ---

        logger.info("Starting EnkiBot polling...")
        # The run method is now part of EnkiBotApplication, or keep polling here
        # For simplicity, keeping polling here:
        ptb_app.run_polling(allowed_updates=Update.ALL_TYPES)
        # Alternatively, if you add a run() method to EnkiBotApplication:
        # enkibot_app_instance.run()
        
        logger.info("EnkiBot has stopped.")
    except Exception as e:
        logger.critical(f"Unrecoverable error during bot setup or run: {e}", exc_info=True)

if __name__ == '__main__':
    main()
