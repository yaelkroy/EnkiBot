# enkibot/core/intent_handlers/general_handler.py
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
from typing import Optional, TYPE_CHECKING

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from enkibot.utils.message_utils import is_forwarded_message, clean_output_text
from enkibot.utils.quota_middleware import enforce_user_quota
from enkibot.utils.text_splitter import split_text_into_chunks

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.modules.response_generator import ResponseGenerator
    from enkibot.modules.fact_check import FactCheckBot

logger = logging.getLogger(__name__)

class GeneralIntentHandler:
    def __init__(self, 
                 language_service: 'LanguageService', 
                 response_generator: 'ResponseGenerator',
                 fact_check_bot: 'FactCheckBot' # Add fact_check_bot here
                 ):
        logger.info("GeneralIntentHandler __init__ STARTING")
        self.language_service = language_service
        self.response_generator = response_generator
        self.fact_check_bot = fact_check_bot
        logger.info("GeneralIntentHandler __init__ COMPLETED")

    async def handle_request(self, 
                             update: Update, 
                             context: ContextTypes.DEFAULT_TYPE, 
                             user_msg_txt: str, 
                             master_intent: str) -> None:
        """
        Handles USER_PROFILE_QUERY, GENERAL_CHAT, and UNKNOWN_INTENT,
        and can serve as a fallback for other intents if needed.
        """
        if not update.message or not update.effective_chat or not update.effective_user:
            logger.warning("GeneralIntentHandler.handle_request called with invalid update/context.")
            return

        logger.info(f"GeneralIntentHandler: Handling intent '{master_intent}' for: '{user_msg_txt[:70]}...'")
        if not await enforce_user_quota(self.response_generator.db_manager, update.effective_user.id, "llm"):
            await update.message.reply_text(self.language_service.get_response_string("llm_quota_exceeded"))
            return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        if is_forwarded_message(update.message):
            forwarded_text = update.message.text or update.message.caption or ""
            
            # Use the fact_check_bot's new method to run a check directly
            await self.fact_check_bot.run_direct_check(update, context, forwarded_text)
            
            # The general handler should not proceed with its own logic for forwarded messages
            return

        main_orchestrator_prompts = self.language_service.get_llm_prompt_set("main_orchestrator")
        system_prompt_override = "You are EnkiBot, a helpful and friendly AI assistant."
        
        if main_orchestrator_prompts and "system" in main_orchestrator_prompts:
            system_prompt_template = main_orchestrator_prompts["system"]
            language_name_for_prompt = self.language_service.current_lang 
            try: 
                from babel import Locale
                locale = Locale.parse(self.language_service.current_lang)
                language_name_for_prompt = locale.get_display_name('en') # Get language name in English
            except ImportError: 
                logger.debug("Babel not installed, using ISO code as language_name in main_orchestrator prompt.")
            except Exception as e: 
                logger.warning(f"Could not get display name for lang {self.language_service.current_lang}: {e}")

            try:
                system_prompt_override = system_prompt_template.format(
                    language_name=language_name_for_prompt, 
                    lang_code=self.language_service.current_lang
                )
            except KeyError as ke:
                logger.error(f"KeyError formatting main_orchestrator system prompt: {ke}. Using template as is: '{system_prompt_template}'")
                system_prompt_override = system_prompt_template # Use unformatted if keys missing
            
            logger.info(f"GeneralIntentHandler: Using system prompt with lang instruction for {self.language_service.current_lang}")
        
        logger.info(f"GeneralIntentHandler: FINAL SYSTEM PROMPT for get_orchestrated_llm_response (lang: {self.language_service.current_lang}): '{system_prompt_override[:100]}...'")
        
        reply = await self.response_generator.get_orchestrated_llm_response(
            prompt_text=user_msg_txt, 
            chat_id=update.effective_chat.id, 
            user_id=update.effective_user.id, 
            message_id=update.message.message_id, 
            context=context, 
            lang_code=self.language_service.current_lang,
            system_prompt_override=system_prompt_override, 
            user_search_ambiguous_response_template=self.language_service.get_response_string("user_search_ambiguous_clarification"),
            user_search_not_found_response_template=self.language_service.get_response_string("user_search_not_found_in_db") 
        )
        
        if reply:
            reply = clean_output_text(reply) or reply

            keyboard = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton("\U0001F504 Regenerate", callback_data="refine:regenerate"),
                    InlineKeyboardButton("\u2795 Expand", callback_data="refine:expand"),
                    InlineKeyboardButton("\U0001F4DD Summarize", callback_data="refine:summary"),
                ]]
            )
            reply_chunks = split_text_into_chunks(reply)
            for idx, chunk in enumerate(reply_chunks):
                await update.message.reply_text(chunk, reply_markup=keyboard if idx == 0 else None)
        else:
            await update.message.reply_text(self.language_service.get_response_string("llm_error_fallback"))
