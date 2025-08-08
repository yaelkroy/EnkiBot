# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
﻿# enkibot/core/telegram_handlers.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)

import logging
import asyncio
import os 
import uuid 
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import Application, ContextTypes, ConversationHandler, CommandHandler, MessageHandler, filters
from telegram.constants import ChatAction
import re 

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.utils.database import DatabaseManager
    from enkibot.core.llm_services import LLMServices
    from enkibot.modules.intent_recognizer import IntentRecognizer
    from enkibot.modules.profile_manager import ProfileManager
    from enkibot.modules.api_router import ApiRouter
    from enkibot.modules.response_generator import ResponseGenerator
    from .intent_handlers.weather_handler import WeatherIntentHandler
    from .intent_handlers.news_handler import NewsIntentHandler
    from .intent_handlers.general_handler import GeneralIntentHandler
    from .intent_handlers.image_generation_handler import ImageGenerationIntentHandler

from enkibot import config as bot_config
from .intent_handlers.weather_handler import WeatherIntentHandler 
from .intent_handlers.news_handler import NewsIntentHandler
from .intent_handlers.general_handler import GeneralIntentHandler
from .intent_handlers.image_generation_handler import ImageGenerationIntentHandler

logger = logging.getLogger(__name__)

# Conversation states
ASK_CITY = 1
ASK_NEWS_TOPIC = 2 

class TelegramHandlerService:
    def __init__(self, 
                 application: Application, 
                 db_manager: 'DatabaseManager', 
                 llm_services: 'LLMServices',   
                 intent_recognizer: 'IntentRecognizer', 
                 profile_manager: 'ProfileManager',     
                 api_router: 'ApiRouter',             
                 response_generator: 'ResponseGenerator', 
                 language_service: 'LanguageService',
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

        # Instantiate specialized handlers
        self.weather_handler = WeatherIntentHandler(
            language_service=self.language_service,
            intent_recognizer=self.intent_recognizer,
            api_router=self.api_router,
            response_generator=self.response_generator,
            pending_action_data_ref=self.pending_action_data
        )
        self.news_handler = NewsIntentHandler(
            language_service=self.language_service,
            intent_recognizer=self.intent_recognizer,
            api_router=self.api_router,
            response_generator=self.response_generator,
            pending_action_data_ref=self.pending_action_data
        )
        self.general_handler = GeneralIntentHandler(
            language_service=self.language_service,
            response_generator=self.response_generator
        )
        self.image_generation_handler = ImageGenerationIntentHandler(
            language_service=self.language_service,
            intent_recognizer=self.intent_recognizer,
            llm_services=self.llm_services
        )
        logger.info("TelegramHandlerService __init__ COMPLETED")

    async def log_message_and_profile_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text or not update.effective_user: 
            return
        chat_id = update.effective_chat.id
        user = update.effective_user
        message = update.message
        if self.allowed_group_ids and chat_id not in self.allowed_group_ids: 
            return 
        current_lang_for_log = self.language_service.current_lang 
        action_taken = await self.db_manager.log_chat_message_and_upsert_user(
            chat_id=chat_id, user_id=user.id, username=user.username,
            first_name=user.first_name, last_name=user.last_name,
            message_id=message.message_id, message_text=message.text,
            preferred_language=current_lang_for_log )
        logger.info(f"Message from user {user.id} logged. Profile action: {action_taken}.")
        name_var_prompts = self.language_service.get_llm_prompt_set("name_variation_generator")
        if action_taken and action_taken.lower() == "insert" and name_var_prompts and "system" in name_var_prompts:
            asyncio.create_task(self.profile_manager.populate_name_variations_with_llm(
                user_id=user.id, first_name=user.first_name, last_name=user.last_name, username=user.username,
                system_prompt=name_var_prompts["system"], user_prompt_template=name_var_prompts.get("user","Generate for: {name_info}")))
        profile_create_prompts = self.language_service.get_llm_prompt_set("profile_creator")
        profile_update_prompts = self.language_service.get_llm_prompt_set("profile_updater")
        if message.text and len(message.text.strip()) > 10:
            if profile_create_prompts and "system" in profile_create_prompts and \
               profile_update_prompts and "system" in profile_update_prompts:
                asyncio.create_task(self.profile_manager.analyze_and_update_user_profile(
                    user_id=user.id, message_text=message.text,
                    create_system_prompt=profile_create_prompts["system"], create_user_prompt_template=profile_create_prompts.get("user","Analyze: {message_text}"),
                    update_system_prompt=profile_update_prompts["system"], update_user_prompt_template=profile_update_prompts.get("user","Update based on: {message_text} with existing: {current_profile_notes}")))
            else: logger.warning(f"Profile prompts missing for lang '{current_lang_for_log}'. Skipping profile analysis for user {user.id}.")

    async def _is_triggered(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt_lower: str) -> bool:
        if not update.message or not context.bot: return False
        current_chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ['group', 'supergroup']
        if not is_group: return True
        if self.allowed_group_ids and current_chat_id not in self.allowed_group_ids: return False
        bot_username_lower = getattr(context.bot, 'username', "").lower() if getattr(context.bot, 'username', None) else ""
        is_at_mentioned = bool(bot_username_lower and f"@{bot_username_lower}" in user_msg_txt_lower)
        is_nickname_mentioned = any(re.search(r'\b' + re.escape(nick.lower()) + r'\b', user_msg_txt_lower, re.I) for nick in self.bot_nicknames)
        is_bot_mentioned = is_at_mentioned or is_nickname_mentioned
        is_reply_to_bot = (update.message.reply_to_message and update.message.reply_to_message.from_user and context.bot and update.message.reply_to_message.from_user.id == context.bot.id)
        final_trigger_decision = is_bot_mentioned or is_reply_to_bot
        if is_group and final_trigger_decision: logger.info(f"_is_triggered: True (Group: {current_chat_id}): @M={is_at_mentioned}, NickM={is_nickname_mentioned}, Reply={is_reply_to_bot}")
        return final_trigger_decision

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.voice:
            return
        
        # --- ADDED THIS CHECK ---
        # Ensure bot only processes voice in allowed groups or private chats
        chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ['group', 'supergroup']
        if is_group and self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            logger.debug(f"handle_voice_message: Skipping voice message from unallowed group {chat_id}.")
            return
        # --- END CHECK ---

        # Set language context to Russian for the response, or detect from user's profile if available
        # For this feature, we will default to Russian for the bot's replies.
        self.language_service._set_current_language_internals('ru')
        
        await update.message.reply_text(self.language_service.get_response_string("voice_message_received"))
        
        voice_file = await update.message.voice.get_file()
        
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.oga")
        
        try:
            await voice_file.download_to_drive(temp_file_path)
            
            transcribed_text = await self.llm_services.transcribe_audio(temp_file_path)
            
            if not transcribed_text:
                await update.message.reply_text(self.language_service.get_response_string("voice_transcription_failed"))
                return
            
            transcription_header = self.language_service.get_response_string("voice_transcription_header")
            await update.message.reply_text(f"*{transcription_header}*\n\n`{transcribed_text}`", parse_mode='MarkdownV2')
            
            is_russian = bool(re.search('[а-яА-Я]', transcribed_text))
            
            if not is_russian:
                logger.info(f"Transcribed text is not Russian. Proceeding with translation.")
                
                translation_prompts = self.language_service.get_llm_prompt_set("text_translator")
                
                if translation_prompts and "system" in translation_prompts and "user_template" in translation_prompts:
                    translated_text = await self.response_generator.translate_text(
                        text_to_translate=transcribed_text,
                        target_language="Russian",
                        system_prompt=translation_prompts["system"],
                        user_prompt_template=translation_prompts["user_template"]
                    )
                    
                    if translated_text:
                        translation_header = self.language_service.get_response_string("voice_translation_header")
                        await update.message.reply_text(f"*{translation_header}*\n\n`{translated_text}`", parse_mode='MarkdownV2')
                    else:
                        logger.error("Translation failed for transcribed text.")
                else:
                    logger.error("Could not find 'text_translator' prompts in language pack.")

        except Exception as e:
            logger.error(f"Error processing voice message: {e}", exc_info=True)
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary audio file: {temp_file_path}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        if not update.message or not update.message.text or not update.effective_chat or not update.effective_user: 
            return None 
        
        await self.language_service.determine_language_context(
            update.message.text, 
            chat_id=update.effective_chat.id,
            update_context=update
        )
        
        await self.log_message_and_profile_tasks(update, context)
        
        chat_id = update.effective_chat.id
        user_msg_txt = update.message.text
        
        current_conv_state = context.user_data.get('conversation_state') if context.user_data else None
        pending_action_details = self.pending_action_data.get(chat_id)

        if current_conv_state == ASK_CITY and pending_action_details and pending_action_details.get("action_type") == "ask_city_weather":
            self.pending_action_data.pop(chat_id, None) 
            context.user_data.pop('conversation_state', None)
            original_msg_id = pending_action_details.get("original_message_id")
            return await self.weather_handler.handle_city_response(update, context, original_msg_id)
        elif current_conv_state == ASK_NEWS_TOPIC and pending_action_details and pending_action_details.get("action_type") == "ask_news_topic":
            self.pending_action_data.pop(chat_id, None) 
            context.user_data.pop('conversation_state', None)
            original_msg_id = pending_action_details.get("original_message_id")
            return await self.news_handler.handle_topic_response(update, context, original_msg_id)
        
        user_msg_txt_lower = user_msg_txt.lower()
        if not await self._is_triggered(update, context, user_msg_txt_lower): 
            return None

        master_intent_prompts = self.language_service.get_llm_prompt_set("master_intent_classifier")
        master_intent = "UNKNOWN_INTENT"
        if master_intent_prompts and "system" in master_intent_prompts:
            user_template_for_master = master_intent_prompts.get("user_template","{text_to_classify}")
            master_intent = await self.intent_recognizer.classify_master_intent(
                text=user_msg_txt, lang_code=self.language_service.current_lang,
                system_prompt=master_intent_prompts["system"], user_prompt_template=user_template_for_master )
        else: 
            logger.error(f"Master intent classification prompt set missing/malformed for lang '{self.language_service.current_lang}'.")
        
        logger.info(f"Master Intent for '{user_msg_txt[:50]}...' (lang: {self.language_service.current_lang}) classified as: {master_intent}")

        next_state: Optional[int] = None
        if master_intent == "WEATHER_QUERY":
            context.user_data['conversation_state'] = ASK_CITY
            next_state = await self.weather_handler.handle_intent(update, context, user_msg_txt)
        elif master_intent == "NEWS_QUERY": 
            context.user_data['conversation_state'] = ASK_NEWS_TOPIC
            next_state = await self.news_handler.handle_intent(update, context, user_msg_txt)
        elif master_intent == "IMAGE_GENERATION_QUERY": 
            await self.image_generation_handler.handle_intent(update, context, user_msg_txt)
            next_state = ConversationHandler.END 
        elif master_intent == "MESSAGE_ANALYSIS_QUERY": 
            await self._handle_message_analysis_query(update, context, user_msg_txt) 
        elif master_intent in ["USER_PROFILE_QUERY", "GENERAL_CHAT", "UNKNOWN_INTENT"]:
            await self.general_handler.handle_request(update, context, user_msg_txt, master_intent)
        else: 
            logger.warning(f"Unhandled master intent type: {master_intent}. Falling back to general handler.")
            await self.general_handler.handle_request(update, context, user_msg_txt, "UNKNOWN_INTENT")
        
        if next_state is None or next_state == ConversationHandler.END:
            if context.user_data and 'conversation_state' in context.user_data:
                context.user_data.pop('conversation_state')
            if chat_id in self.pending_action_data:
                 self.pending_action_data.pop(chat_id, None)
        return next_state

    async def _handle_message_analysis_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> None:
        is_valid_reply_scenario = (
            update.message and update.message.reply_to_message and 
            update.message.reply_to_message.text and 
            update.message.reply_to_message.from_user and context.bot and 
            update.message.reply_to_message.from_user.id != context.bot.id )
        if not is_valid_reply_scenario:
            logger.info("MESSAGE_ANALYSIS_QUERY classified, but not valid reply. Delegating to GeneralIntentHandler.")
            await self.general_handler.handle_request(update, context, user_msg_txt, "GENERAL_CHAT")
            return
        
        logger.info("TelegramHandlers: Processing MESSAGE_ANALYSIS_QUERY directly.")
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
        if not (analyzer_prompts and "system" in analyzer_prompts) : 
            logger.error("Prompt set for replied message analysis is missing or malformed."); 
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message")); 
            return 
        
        analysis_result = await self.response_generator.analyze_replied_message(
            original_text=original_text, user_question=question_for_analysis,
            system_prompt=analyzer_prompts["system"], user_prompt_template=analyzer_prompts.get("user_template") )
        await update.message.reply_text(analysis_result)
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message: return
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE) 
        await update.message.reply_html(self.language_service.get_response_string("start", user_mention=update.effective_user.mention_html()))

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message: return
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE)
        await update.message.reply_text(self.language_service.get_response_string("help"))

    async def news_command_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]: 
        if not update.message or not update.effective_user: return ConversationHandler.END
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE)
        context.user_data['conversation_state'] = ASK_NEWS_TOPIC
        return await self.news_handler.handle_command_entry(update, context)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f'Update "{update}" caused error "{context.error}"', exc_info=True)
        if isinstance(update, Update) and update.effective_chat:
            try:
                error_msg = self.language_service.get_response_string("generic_error_message", "Oops! Something went very wrong on my end.")
                await context.bot.send_message(chat_id=update.effective_chat.id, text=error_msg)
            except Exception as e: 
                logger.error(f"CRITICAL: Error sending error message to user: {e}", exc_info=True)
    
    async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        chat_id = update.effective_chat.id
        if chat_id in self.pending_action_data:
            del self.pending_action_data[chat_id]
        if context.user_data and 'conversation_state' in context.user_data:
            context.user_data.pop('conversation_state')
            
        logger.info(f"User {update.effective_user.id if update.effective_user else ''} cancelled conversation.")
        if update.message: 
            await update.message.reply_text(
                self.language_service.get_response_string("conversation_cancelled", "Okay, current operation cancelled."), 
                reply_markup=ReplyKeyboardRemove() 
            )
        return ConversationHandler.END

    def register_all_handlers(self): 
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message),
                CommandHandler("news", self.news_command_entry) 
            ],
            states={
                ASK_CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)],
                ASK_NEWS_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)],
            },
            fallbacks=[CommandHandler("cancel", self.cancel_conversation)],
            allow_reentry=True 
        )
        self.application.add_handler(conv_handler)

        # Add the standalone handler for voice messages
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
        
        self.application.add_error_handler(self.error_handler)
        logger.info("TelegramHandlerService: All handlers registered.")
