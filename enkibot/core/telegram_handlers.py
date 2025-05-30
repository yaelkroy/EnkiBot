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

# enkibot/core/telegram_handlers.py
import logging
import asyncio
from typing import Optional, Dict, Any, List 

from telegram import Update
from telegram.ext import Application, ContextTypes, ConversationHandler, CommandHandler, MessageHandler, filters
from telegram.constants import ChatAction
import re 

# Import config directly for constants like BOT_NICKNAMES_TO_CHECK, ALLOWED_GROUP_IDS
# This is acceptable if these are truly static configurations.
# Alternatively, they are passed in __init__ as `self.allowed_group_ids` and `self.bot_nicknames`
# The __init__ signature I provided in msg #30 and used in app.py in #28 *does* pass them.
# So, we should use `self.allowed_group_ids` and `self.bot_nicknames`.

from enkibot.core.language_service import LanguageService
from enkibot.utils.database import DatabaseManager # For type hinting if needed for self.db_manager
from enkibot.core.llm_services import LLMServices # For type hinting if needed
from enkibot.modules.intent_recognizer import IntentRecognizer # For type hinting
from enkibot.modules.profile_manager import ProfileManager # For type hinting
from enkibot.modules.api_router import ApiRouter # For type hinting
from enkibot.modules.response_generator import ResponseGenerator # For type hinting


logger = logging.getLogger(__name__)
ASK_CITY = 1 

class TelegramHandlerService:
    def __init__(self, 
                 application: Application, 
                 db_manager: DatabaseManager, 
                 llm_services: LLMServices,   
                 intent_recognizer: IntentRecognizer, 
                 profile_manager: ProfileManager,     
                 api_router: ApiRouter,             
                 response_generator: ResponseGenerator, 
                 language_service: LanguageService,
                 allowed_group_ids: set, 
                 bot_nicknames: list
                ):
        logger.info("TelegramHandlerService __init__ STARTING")
        self.application = application
        self.db_manager = db_manager
        self.llm_services = llm_services 
        self.intent_recognizer = intent_recognizer
        self.profile_manager = profile_manager
        self.api_router = api_router
        self.response_generator = response_generator
        self.language_service = language_service
        
        self.allowed_group_ids = allowed_group_ids 
        self.bot_nicknames = bot_nicknames 
        
        self.pending_action_data: Dict[int, Dict[str, Any]] = {}
        logger.info("TelegramHandlerService __init__ COMPLETED")

    async def log_message_and_profile_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text or not update.effective_user: 
            logger.debug("log_message_and_profile_tasks: Skipping.")
            return
        
        chat_id = update.effective_chat.id
        user = update.effective_user
        message = update.message

        # Use self.allowed_group_ids for filtering here as well
        if self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            logger.debug(f"log_message_and_profile_tasks: Skipping msg from chat {chat_id} (not in allowed group IDs: {self.allowed_group_ids}).")
            return

        current_lang_for_log = self.language_service.current_lang 

        action_taken = await self.db_manager.log_chat_message_and_upsert_user(
            chat_id=chat_id, user_id=user.id, username=user.username,
            first_name=user.first_name, last_name=user.last_name,
            message_id=message.message_id, message_text=message.text,
            preferred_language=current_lang_for_log 
        )
        logger.info(f"Message from user {user.id} logged. Profile action: {action_taken}.")

        name_var_prompts = self.language_service.get_llm_prompt_set("name_variation_generator")
        if action_taken and action_taken.lower() == "insert" and name_var_prompts:
            logger.info(f"New user {user.id} ({user.first_name}). Queuing name variation generation.")
            asyncio.create_task(self.profile_manager.populate_name_variations_with_llm(
                user_id=user.id, first_name=user.first_name, last_name=user.last_name, username=user.username,
                system_prompt=name_var_prompts["system"], 
                user_prompt_template=name_var_prompts.get("user","Generate for: {name_info}")
            ))

        profile_create_prompts = self.language_service.get_llm_prompt_set("profile_creator")
        profile_update_prompts = self.language_service.get_llm_prompt_set("profile_updater")
        if message.text and len(message.text.strip()) > 10:
            if profile_create_prompts and profile_update_prompts:
                logger.info(f"Message from user {user.id} meets criteria for profile analysis. Queuing task.")
                asyncio.create_task(self.profile_manager.analyze_and_update_user_profile(
                    user_id=user.id, message_text=message.text,
                    create_system_prompt=profile_create_prompts["system"],
                    create_user_prompt_template=profile_create_prompts.get("user","Analyze: {message_text}"),
                    update_system_prompt=profile_update_prompts["system"],
                    update_user_prompt_template=profile_update_prompts.get("user","Update based on: {message_text} with existing: {current_profile_notes}")
                ))
            else: 
                logger.warning(f"Profile prompts missing for lang '{current_lang_for_log}'. Skipping profile analysis for user {user.id}.")

    async def _is_triggered(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt_lower: str) -> bool:
        if not update.message or not context.bot:
            logger.debug("_is_triggered: False (no message or bot context)")
            return False
        
        current_chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ['group', 'supergroup']

        if not is_group:
            logger.debug(f"_is_triggered: True (private chat with user {update.effective_user.id if update.effective_user else 'Unknown'})")
            return True

        if self.allowed_group_ids: 
            if current_chat_id not in self.allowed_group_ids:
                logger.info(f"_is_triggered: False (group {current_chat_id} not in self.allowed_group_ids: {self.allowed_group_ids})")
                return False
        
        bot_username_lower = getattr(context.bot, 'username', "").lower() if getattr(context.bot, 'username', None) else ""
        is_at_mentioned = bool(bot_username_lower and f"@{bot_username_lower}" in user_msg_txt_lower)
        
        is_nickname_mentioned = False
        for nick in self.bot_nicknames: 
            if re.search(r'\b' + re.escape(nick.lower()) + r'\b', user_msg_txt_lower, re.I):
                is_nickname_mentioned = True; break
        
        is_bot_mentioned = is_at_mentioned or is_nickname_mentioned
        is_reply_to_bot = (
            update.message.reply_to_message and 
            update.message.reply_to_message.from_user and
            update.message.reply_to_message.from_user.id == context.bot.id
        )
        
        final_trigger_decision = is_bot_mentioned or is_reply_to_bot
        
        if final_trigger_decision:
            logger.info(
                f"_is_triggered: True (Group: {current_chat_id}): "
                f"Text='{user_msg_txt_lower[:30]}...' @Mention={is_at_mentioned}, NickMention={is_nickname_mentioned}, "
                f"ReplyToBot={is_reply_to_bot}"
            )
        # else: # This log can be noisy if not triggered
            # logger.info(
            #      f"_is_triggered: False (Group: {current_chat_id}): No direct interaction. "
            #      f"Text='{user_msg_txt_lower[:30]}...' @Mention={is_at_mentioned}, NickMention={is_nickname_mentioned}, ReplyToBot={is_reply_to_bot}"
            # )
        return final_trigger_decision

    # --- THESE ARE THE MISSING METHODS (or misnamed/misplaced in your local file) ---
    async def _handle_weather_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> Optional[int]:
        logger.info(f"TelegramHandlers: Handling WEATHER_QUERY: '{user_msg_txt}'")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        weather_analysis_prompts = self.language_service.get_llm_prompt_set("weather_intent_analyzer")
        location_extract_prompts = self.language_service.get_llm_prompt_set("location_extractor")

        if not (weather_analysis_prompts and "system" in weather_analysis_prompts and \
                location_extract_prompts and "system" in location_extract_prompts):
            logger.error("Weather intent prompts are missing or malformed.")
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
            return None 
        
        intent_data = await self.intent_recognizer.analyze_weather_request_with_llm(
            text=user_msg_txt, lang_code=self.language_service.current_lang, 
            system_prompt=weather_analysis_prompts["system"], user_prompt_template=weather_analysis_prompts.get("user") )
        city = await self.intent_recognizer.extract_location_with_llm(
            text=user_msg_txt, lang_code=self.language_service.current_lang,
            system_prompt=location_extract_prompts["system"], user_prompt_template=location_extract_prompts.get("user") )
        
        if city:
            weather_report = await self.api_router.get_weather_data(
                location=city, forecast_type=intent_data.get("type", "current"), 
                days=intent_data.get("days", 7), lang_pack_full=self.language_service.current_lang_pack_full )
            await update.message.reply_text(weather_report); return None 
        else:
            self.pending_action_data[update.effective_chat.id] = { 
                "action_type": "ask_city_weather", "forecast_type": intent_data.get("type", "current"), 
                "days": intent_data.get("days", 7), "original_message_id": update.message.message_id }
            await update.message.reply_text(self.language_service.get_response_string("weather_ask_city")); return ASK_CITY

    async def _handle_news_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> None:
        logger.info(f"TelegramHandlers: Handling NEWS_QUERY: '{user_msg_txt}'")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        news_topic_prompts = self.language_service.get_llm_prompt_set("news_topic_extractor")
        if not (news_topic_prompts and "system" in news_topic_prompts): 
            logger.error("News topic extraction prompt is missing or malformed."); 
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message")); return
        
        topic = await self.intent_recognizer.extract_news_topic_with_llm(
            text=user_msg_txt, lang_code=self.language_service.current_lang, 
            system_prompt=news_topic_prompts["system"], user_prompt_template=news_topic_prompts.get("user") )
        news_report = await self.api_router.get_latest_news( 
            query=topic, lang_code=self.language_service.current_lang, 
            response_strings=self.language_service.current_response_strings )
        await update.message.reply_text(news_report, disable_web_page_preview=True)

    async def _handle_message_analysis_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> bool:
        if not (update.message and update.message.reply_to_message and 
                update.message.reply_to_message.text and 
                update.message.reply_to_message.from_user and
                context.bot and update.message.reply_to_message.from_user.id != context.bot.id):
            logger.debug("_handle_message_analysis_query: Conditions not met for reply analysis.")
            return False # Not a valid scenario for this handler to take over
        
        logger.info("TelegramHandlers: Processing MESSAGE_ANALYSIS_QUERY.")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        original_text = update.message.reply_to_message.text
        question_for_analysis = user_msg_txt
        
        bot_username_lower = getattr(context.bot, 'username', "").lower() if getattr(context.bot, 'username', None) else ""
        cleaned_question = user_msg_txt.lower()
        for nick in self.bot_nicknames + ([f"@{bot_username_lower}"] if bot_username_lower else []):
            cleaned_question = cleaned_question.replace(nick.lower(), "").strip()
        if len(cleaned_question) < 5: 
            question_for_analysis = self.language_service.get_response_string("replied_message_default_question")
        
        analyzer_prompts = self.language_service.get_llm_prompt_set("replied_message_analyzer")
        if not (analyzer_prompts and "system" in analyzer_prompts): 
            logger.error("Prompt set for replied message analysis is missing or malformed."); 
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message")); 
            return True # Error handled by message
        
        analysis_result = await self.response_generator.analyze_replied_message(
            original_text=original_text, user_question=question_for_analysis,
            system_prompt=analyzer_prompts["system"], user_prompt_template=analyzer_prompts.get("user") )
        await update.message.reply_text(analysis_result)
        return True # Handled

    async def _handle_user_profile_query_or_general_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str, master_intent: str) -> None:
        logger.info(f"TelegramHandlers: Handling {master_intent}: '{user_msg_txt[:50]}...'")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        main_orchestrator_prompts = self.language_service.get_llm_prompt_set("main_orchestrator")
        system_prompt_override = (main_orchestrator_prompts["system"] 
                                  if main_orchestrator_prompts and "system" in main_orchestrator_prompts
                                  else "You are EnkiBot, a helpful AI assistant.")
        
        reply = await self.response_generator.get_orchestrated_llm_response(
            prompt_text=user_msg_txt, chat_id=update.effective_chat.id, user_id=update.effective_user.id, 
            message_id=update.message.message_id, context=context, lang_code=self.language_service.current_lang,
            system_prompt_override=system_prompt_override, 
            user_search_ambiguous_response_template=self.language_service.get_response_string("user_search_ambiguous_clarification"),
            user_search_not_found_response_template=self.language_service.get_response_string("user_search_not_found_in_db") )
        
        if reply: await update.message.reply_text(reply)
        else: await update.message.reply_text(self.language_service.get_response_string("llm_error_fallback"))
    # --- END OF MISSING METHODS ---

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        if not update.message or not update.message.text or not update.effective_chat or not update.effective_user: 
            return None 
        
        await self.language_service.determine_language_context(
            update.message.text, 
            chat_id=update.effective_chat.id,
            update_context=update
        )
        logger.info(f"TelegramHandlers: Lang set to: {self.language_service.current_lang} for user {update.effective_user.id}")
        
        await self.log_message_and_profile_tasks(update, context)
        
        pending_action = self.pending_action_data.pop(update.effective_chat.id, None)
        if pending_action and pending_action["action_type"] == "ask_city_weather":
            city_name = update.message.text
            logger.info(f"Received city '{city_name}' for pending weather request.")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            weather_report = await self.api_router.get_weather_data( 
                location=city_name, forecast_type=pending_action["forecast_type"], 
                days=pending_action["days"], lang_pack_full=self.language_service.current_lang_pack_full )
            await update.message.reply_text(weather_report, reply_to_message_id=pending_action["original_message_id"])
            return ConversationHandler.END 

        user_msg_txt = update.message.text
        user_msg_txt_lower = user_msg_txt.lower()

        if not await self._is_triggered(update, context, user_msg_txt_lower): 
            return None

        master_intent_prompts = self.language_service.get_llm_prompt_set("master_intent_classifier")
        master_intent = "UNKNOWN_INTENT"
        if master_intent_prompts and "system" in master_intent_prompts:
            user_template_for_master = master_intent_prompts.get("user","{text_to_classify}")
            master_intent = await self.intent_recognizer.classify_master_intent(
                text=user_msg_txt, lang_code=self.language_service.current_lang,
                system_prompt=master_intent_prompts["system"], user_prompt_template=user_template_for_master )
        else: 
            logger.error(f"Master intent classification prompt set missing/malformed for lang '{self.language_service.current_lang}'.")
        
        logger.info(f"Master Intent for '{user_msg_txt[:50]}...' classified as: {master_intent}")

        if master_intent == "WEATHER_QUERY":
            state = await self._handle_weather_intent(update, context, user_msg_txt) # This line was erroring
            return state 
        elif master_intent == "NEWS_QUERY": 
            await self._handle_news_intent(update, context, user_msg_txt)
        elif master_intent == "MESSAGE_ANALYSIS_QUERY": 
            if not (update.message.reply_to_message and \
                    update.message.reply_to_message.text and \
                    update.message.reply_to_message.from_user and \
                    context.bot and \
                    update.message.reply_to_message.from_user.id != context.bot.id):
                logger.info("MESSAGE_ANALYSIS_QUERY classified, but not a valid reply-to-another. Treating as GENERAL_CHAT.")
                await self._handle_user_profile_query_or_general_chat(update, context, user_msg_txt, "GENERAL_CHAT")
            else:
                await self._handle_message_analysis_query(update, context, user_msg_txt)
        elif master_intent in ["USER_PROFILE_QUERY", "GENERAL_CHAT", "UNKNOWN_INTENT"]:
            await self._handle_user_profile_query_or_general_chat(update, context, user_msg_txt, master_intent)
        else: 
            logger.warning(f"Unhandled master intent type: {master_intent}. Falling back to general.")
            await self._handle_user_profile_query_or_general_chat(update, context, user_msg_txt, "UNKNOWN_INTENT")
        return None
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message: return
        self.language_service._set_current_language_internals(config.DEFAULT_LANGUAGE) 
        await update.message.reply_html(self.language_service.get_response_string("start", user_mention=update.effective_user.mention_html()))

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        self.language_service._set_current_language_internals(config.DEFAULT_LANGUAGE)
        await update.message.reply_text(self.language_service.get_response_string("help"))

    async def news_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat: return
        self.language_service._set_current_language_internals(config.DEFAULT_LANGUAGE)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        report = await self.api_router.get_latest_news(
            query=None, lang_code=self.language_service.current_lang, 
            response_strings=self.language_service.current_response_strings )
        await update.message.reply_text(report, disable_web_page_preview=True)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f'Update "{update}" caused error "{context.error}"', exc_info=True)
        if isinstance(update, Update) and update.effective_chat:
            try:
                error_msg = self.language_service.get_response_string(
                    "generic_error_message", 
                    "Oops! Something went very wrong on my end. I've logged the issue."
                )
                await context.bot.send_message(chat_id=update.effective_chat.id, text=error_msg)
            except Exception as e: 
                logger.error(f"CRITICAL: Error sending error message to user: {e}", exc_info=True)
    
    def register_all_handlers(self): 
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("news", self.news_command))
        
        conv_handler = ConversationHandler(
            entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)],
            states={ ASK_CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)],},
            fallbacks=[], 
            allow_reentry=True 
        )
        self.application.add_handler(conv_handler)
        self.application.add_error_handler(self.error_handler)
        logger.info("TelegramHandlerService: All handlers registered.")