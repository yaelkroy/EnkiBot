# enkibot/app.py
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
import logging
from telegram import Update
from telegram.ext import Application

from enkibot import config
from enkibot.utils.database import DatabaseManager
from enkibot.core.llm_services import LLMServices
from enkibot.core.language_service import LanguageService
from enkibot.modules.intent_recognizer import IntentRecognizer
from enkibot.modules.profile_manager import ProfileManager
from enkibot.modules.api_router import ApiRouter
from enkibot.modules.response_generator import ResponseGenerator
from enkibot.core.telegram_handlers import TelegramHandlerService
from enkibot.modules.karma_manager import KarmaManager 

logger = logging.getLogger(__name__)

class EnkiBotApplication:
    def __init__(self, ptb_application: Application):
        logger.info("EnkiBotApplication initializing...")
        self.ptb_application = ptb_application

        # Initialize core services
        self.db_manager = DatabaseManager(config.DB_CONNECTION_STRING)
        self.llm_services = LLMServices(
            openai_api_key=config.OPENAI_API_KEY, openai_model_id=config.OPENAI_MODEL_ID,
            groq_api_key=config.GROQ_API_KEY, groq_model_id=config.GROQ_MODEL_ID, groq_endpoint_url=config.GROQ_ENDPOINT_URL,
            openrouter_api_key=config.OPENROUTER_API_KEY, openrouter_model_id=config.OPENROUTER_MODEL_ID, openrouter_endpoint_url=config.OPENROUTER_ENDPOINT_URL,
            google_ai_api_key=config.GOOGLE_AI_API_KEY, google_ai_model_id=config.GOOGLE_AI_MODEL_ID
        )
        self.language_service = LanguageService(
            llm_services=self.llm_services, 
            db_manager=self.db_manager # Pass db_manager for fetching chat history
        )
        
        # Initialize functional modules/services
        self.intent_recognizer = IntentRecognizer(self.llm_services)
        self.profile_manager = ProfileManager(self.llm_services, self.db_manager)
        self.api_router = ApiRouter(
            weather_api_key=config.WEATHER_API_KEY, 
            news_api_key=config.NEWS_API_KEY,
            llm_services=self.llm_services
        )
        self.response_generator = ResponseGenerator(
            self.llm_services, 
            self.db_manager, 
            self.intent_recognizer
        )

        # Initialize Telegram handlers, passing all necessary services
        self.handler_service = TelegramHandlerService(
            application=self.ptb_application,
            db_manager=self.db_manager,
            llm_services=self.llm_services,
            intent_recognizer=self.intent_recognizer,
            profile_manager=self.profile_manager,
            api_router=self.api_router,
            response_generator=self.response_generator,
            language_service=self.language_service,
            allowed_group_ids=config.ALLOWED_GROUP_IDS, # Pass as set
            bot_nicknames=config.BOT_NICKNAMES_TO_CHECK # Pass as list
        )
        
        logger.info("EnkiBotApplication initialized all services.")

    def register_handlers(self):
        """Registers all Telegram handlers."""
        self.handler_service.register_all_handlers()
        logger.info("EnkiBotApplication: All handlers registered with PTB Application.")

    def run(self):
        """Starts the bot polling."""
        logger.info("EnkiBotApplication: Starting PTB Application polling...")
        self.ptb_application.run_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("EnkiBotApplication: Polling stopped.")
