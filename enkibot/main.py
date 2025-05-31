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

import os
import logging
import asyncio
from typing import Optional 
from telegram import Update 
from telegram.ext import Application 

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
        ptb_app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        
        logger.info("Initializing EnkiBotApplication...")
        # --- MODIFIED BOT INSTANTIATION ---
        enkibot_app_instance = EnkiBotApplication(ptb_app) 
        enkibot_app_instance.register_handlers() # Call the method to register handlers
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