
# enkibot/core/intent_handlers/weather_handler.py
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
from typing import Optional, Dict, Any, TYPE_CHECKING
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler
from telegram.constants import ChatAction
from enkibot.utils.quota_middleware import enforce_user_quota

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.modules.intent_recognizer import IntentRecognizer
    from enkibot.modules.api_router import ApiRouter
    from enkibot.modules.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)
ASK_CITY = 1 # Define state here or import from a central states definition

class WeatherIntentHandler:
    def __init__(self, 
                 language_service: 'LanguageService', 
                 intent_recognizer: 'IntentRecognizer',
                 api_router: 'ApiRouter',
                 response_generator: 'ResponseGenerator',
                 pending_action_data_ref: Dict[int, Dict[str, Any]]): # Reference to the main pending_action_data
        self.language_service = language_service
        self.intent_recognizer = intent_recognizer
        self.api_router = api_router
        self.response_generator = response_generator
        self.pending_action_data = pending_action_data_ref # Use the shared dictionary

    async def _process_weather_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, city: str, original_message_id: Optional[int] = None) -> int:
        # This is the _process_weather_request method moved from TelegramHandlerService
        if not update.message or not update.effective_chat: return ConversationHandler.END
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        logger.info(f"WeatherHandler: Processing weather for city: '{city}'")
        
        forecast_data = await self.api_router.get_weather_data_structured(
            location=city, 
            lang_pack_full=self.language_service.current_lang_pack_full
        )
        reply_to_id = original_message_id or update.message.message_id
        if forecast_data:
            compiler_prompts = self.language_service.get_llm_prompt_set("weather_forecast_compiler")
            if not (compiler_prompts and "system" in compiler_prompts and "user_template" in compiler_prompts):
                logger.error("Weather forecast compiler LLM prompts are missing.")
                await update.message.reply_text(self.language_service.get_response_string("weather_api_data_error", location=city), reply_to_message_id=reply_to_id)
            else:
                if not await enforce_user_quota(self.response_generator.db_manager, update.effective_user.id, "llm"):
                    await update.message.reply_text(self.language_service.get_response_string("llm_quota_exceeded"))
                    return ConversationHandler.END
                compiled_response = await self.response_generator.compile_weather_forecast_response(
                    forecast_data_structured=forecast_data,
                    lang_code=self.language_service.current_lang,
                    system_prompt=compiler_prompts["system"],
                    user_prompt_template=compiler_prompts["user_template"]
                )
                await update.message.reply_text(compiled_response, reply_to_message_id=reply_to_id)
        else:
            await update.message.reply_text(self.language_service.get_response_string("weather_city_not_found", location=city), reply_to_message_id=reply_to_id)
        return ConversationHandler.END

    async def handle_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> Optional[int]:
        # This is the _handle_weather_intent method logic
        logger.info(f"WeatherHandler: Initial WEATHER_QUERY: '{user_msg_txt}'")
        if not update.message or not update.effective_chat: return ConversationHandler.END

        if not await enforce_user_quota(self.response_generator.db_manager, update.effective_user.id, "llm"):
            await update.message.reply_text(self.language_service.get_response_string("llm_quota_exceeded"))
            return ConversationHandler.END

        location_extract_prompts = self.language_service.get_llm_prompt_set("location_extractor")
        if not (location_extract_prompts and "system" in location_extract_prompts):
            logger.error("Location extractor LLM prompts missing.")
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
            return ConversationHandler.END

        city = await self.intent_recognizer.extract_location_with_llm(
            text=user_msg_txt, lang_code=self.language_service.current_lang,
            system_prompt=location_extract_prompts["system"], 
            user_prompt_template=location_extract_prompts.get("user","{text}")
        )
        if city: # This will be false because extract_location_with_llm returns None
            return await self._process_weather_request(update, context, city, update.message.message_id)
        else:
            logger.info("WeatherHandler: No city identified, asking user.")
            self.pending_action_data[update.effective_chat.id] = {
                "action_type": "ask_city_weather", 
                "original_message_id": update.message.message_id 
            }
            await update.message.reply_text(self.language_service.get_response_string("weather_ask_city"))
            return ASK_CITY 

    async def handle_city_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE, original_message_id: Optional[int]) -> int:
        """
        Handles the user's response when they provide a city name after being asked.
        """
        if not update.message or not update.message.text:
            await update.message.reply_text(self.language_service.get_response_string("weather_ask_city"))
            return ASK_CITY # Re-ask

        user_reply_city_text = update.message.text.strip()
        logger.info(f"WeatherHandler: Received city reply '{user_reply_city_text}' in ASK_CITY state.")

        extractor_prompts = self.language_service.get_llm_prompt_set("location_reply_extractor")
        if not (extractor_prompts and "system" in extractor_prompts and "user_template" in extractor_prompts):
            logger.error("Location reply extractor LLM prompts are missing or malformed.")
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
            return ConversationHandler.END

        if not await enforce_user_quota(self.response_generator.db_manager, update.effective_user.id, "llm"):
            await update.message.reply_text(self.language_service.get_response_string("llm_quota_exceeded"))
            return ConversationHandler.END

        extracted_city = await self.intent_recognizer.extract_location_from_reply(
            text=user_reply_city_text,
            lang_code=self.language_service.current_lang,
            system_prompt=extractor_prompts["system"],
            user_prompt_template=extractor_prompts["user_template"]
        )

        if extracted_city:
            return await self._process_weather_request(update, context, extracted_city, original_message_id)
        else:
            # LLM couldn't extract a clear city from the reply
            await update.message.reply_text(self.language_service.get_response_string("weather_ask_city_failed_extraction", 
                                                                                      "Sorry, I didn't quite catch the city name. Could you please tell me the city again?"))
            return ASK_CITY # Remain in the same state, re-prompting
