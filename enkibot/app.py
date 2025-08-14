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
from datetime import timedelta, time as dtime
from telegram import Update
from telegram.ext import Application, ContextTypes

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
from enkibot.modules.spam_detector import SpamDetector
from enkibot.modules.stats_manager import StatsManager
from enkibot.modules.community_moderation import CommunityModerationService
from enkibot.modules.fact_check import (
    FactChecker,
    FactCheckBot,
    SatireDetector,
    StanceModel,
    OpenAIWebFetcher,
)
from enkibot.modules.primary_source_hunter import PrimarySourceHunter

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
            db_manager=self.db_manager
        )
        
        # Initialize functional modules/services
        self.intent_recognizer = IntentRecognizer(self.llm_services)
        self.profile_manager = ProfileManager(self.llm_services, self.db_manager)
        self.api_router = ApiRouter(
            weather_api_key=config.WEATHER_API_KEY,
            news_api_key=config.NEWS_API_KEY,
            llm_services=self.llm_services,
            db_manager=self.db_manager,
        )
        self.response_generator = ResponseGenerator(
            self.llm_services,
            self.db_manager,
            self.intent_recognizer
        )
        self.spam_detector = SpamDetector(
            self.llm_services,
            self.db_manager,
            self.language_service,
            enabled=config.ENABLE_SPAM_DETECTION,
        )
        self.stats_manager = StatsManager(self.db_manager)
        self.karma_manager = KarmaManager(self.db_manager)
        self.community_moderation = CommunityModerationService(
            self.language_service,
            admin_chat_id=config.REPORTS_CHANNEL_ID,
        )

        # ------------------------------------------------------------------
        # Fact checking subsystem (skeleton implementation)
        # ------------------------------------------------------------------
        self.fact_checker = FactChecker(
            fetcher=OpenAIWebFetcher(),
            stance=StanceModel(),
            llm_services=self.llm_services,
            primary_hunter=PrimarySourceHunter(),
        )

        def _default_fact_cfg(_chat_id: int) -> dict:
            return {"satire": {"enabled": False}, "auto": {"auto_check_news": True}}

        self.fact_check_bot = FactCheckBot(
            app=self.ptb_application,
            fc=self.fact_checker,
            satire_detector=SatireDetector(_default_fact_cfg),
            cfg_reader=_default_fact_cfg,
            db_manager=self.db_manager,
            language_service=self.language_service,
        )
        self.fact_check_bot.register()

        # Update GeneralIntentHandler to include the fact_check_bot instance
        self.general_handler = GeneralIntentHandler(
            language_service=self.language_service,
            response_generator=self.response_generator,
            fact_check_bot=self.fact_check_bot,
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
            spam_detector=self.spam_detector,
            stats_manager=self.stats_manager,
            karma_manager=self.karma_manager,
            community_moderation=self.community_moderation,
            allowed_group_ids=config.ALLOWED_GROUP_IDS,
            bot_nicknames=config.BOT_NICKNAMES_TO_CHECK,
            # Pass the instance of the new handler, not the class
            general_handler=self.general_handler,
        )

        self.spam_detector.set_captcha_callback(
            self.handler_service.start_captcha
        )

        async def _refresh_news_channels_job(_: ContextTypes.DEFAULT_TYPE) -> None:
            logger.info("News channel refresh job started")
            await self.db_manager.refresh_news_channels()
            logger.info("News channel refresh job completed")

        if getattr(self.ptb_application, "job_queue", None):
            logger.info("Scheduling news channel refresh job")
            self.ptb_application.job_queue.run_once(
                _refresh_news_channels_job, when=0
            )
            self.ptb_application.job_queue.run_monthly(
                _refresh_news_channels_job,
                when=dtime(hour=0, minute=0),
                day=1,
            )
        else:
            logger.warning(
                "JobQueue is not available. Running one-off news channel refresh."
            )
            import asyncio

            asyncio.run(self.db_manager.refresh_news_channels())

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
