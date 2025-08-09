# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# enkibot/core/telegram_handlers.py
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

import logging
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from types import SimpleNamespace

from telegram import Update, ReplyKeyboardRemove, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ContextTypes, ConversationHandler, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ChatAction
import re
import random

try:
    from nudenet import NudeClassifier
except Exception:
    NudeClassifier = None

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.utils.database import DatabaseManager
    from enkibot.core.llm_services import LLMServices
    from enkibot.modules.intent_recognizer import IntentRecognizer
    from enkibot.modules.profile_manager import ProfileManager
    from enkibot.modules.api_router import ApiRouter
    from enkibot.modules.response_generator import ResponseGenerator
    from enkibot.modules.spam_detector import SpamDetector
    from enkibot.modules.stats_manager import StatsManager
    from .intent_handlers.weather_handler import WeatherIntentHandler
    from .intent_handlers.news_handler import NewsIntentHandler
    from .intent_handlers.general_handler import GeneralIntentHandler
    from .intent_handlers.image_generation_handler import ImageGenerationIntentHandler

from enkibot import config as bot_config
from .intent_handlers.weather_handler import WeatherIntentHandler 
from .intent_handlers.news_handler import NewsIntentHandler
from .intent_handlers.general_handler import GeneralIntentHandler
from .intent_handlers.image_generation_handler import ImageGenerationIntentHandler
from enkibot.modules.spam_detector import SpamDetector

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
                 spam_detector: 'SpamDetector',
                 stats_manager: 'StatsManager',
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
        self.spam_detector = spam_detector
        self.stats_manager = stats_manager
        
        self.allowed_group_ids = allowed_group_ids 
        self.bot_nicknames = bot_nicknames

        self.pending_action_data: Dict[int, Dict[str, Any]] = {}
        self.pending_captchas: Dict[int, Dict[str, Any]] = {}
        self.nsfw_classifier: Optional['NudeClassifier'] = None

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
        await self.stats_manager.log_message(chat_id, user.id, message.text, user.username)
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

    def _extract_ai_trigger(self, text: str) -> tuple[str, bool]:
        """Detects natural language invocations like 'Hey, Enki!' and strips them."""
        pattern = r'^\s*hey[\s,]+enki[!,:]*\s*'
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            cleaned = text[match.end():].lstrip()
            return cleaned, True
        return text, False

    async def _is_user_admin(self, chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
        try:
            member = await context.bot.get_chat_member(chat_id, user_id)
            return member.status in ("administrator", "creator")
        except Exception:
            return False

    def _generate_math_captcha(self) -> tuple[str, list[int], int]:
        a, b = random.randint(1, 20), random.randint(1, 20)
        question = f"{a} + {b}"
        correct = a + b
        options = {correct}
        while len(options) < 4:
            options.add(random.randint(correct - 10, correct + 10))
        option_list = list(options)
        random.shuffle(option_list)
        return question, option_list, correct

    async def _start_captcha(self, user, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
        method = random.choice(["button", "math"])
        mention = user.mention_html()
        if method == "button":
            text = (
                f"Welcome, {mention}! Please prove you are human by clicking the button below within {bot_config.CAPTCHA_TIMEOUT_SECONDS} seconds."
            )
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("✅ I'm a human!", callback_data=f"captcha_button:{user.id}")]]
            )
            msg = await context.bot.send_message(chat_id, text, reply_markup=keyboard, parse_mode="HTML")
            self.pending_captchas[user.id] = {
                "chat_id": chat_id,
                "message_id": msg.message_id,
                "type": "button",
                "mention": mention,
            }
        else:
            question, options, correct = self._generate_math_captcha()
            text = (
                f"Welcome, {mention}! Solve: {question} = ?"
            )
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton(str(opt), callback_data=f"captcha_math:{user.id}:{opt}") for opt in options]]
            )
            msg = await context.bot.send_message(chat_id, text, reply_markup=keyboard, parse_mode="HTML")
            self.pending_captchas[user.id] = {
                "chat_id": chat_id,
                "message_id": msg.message_id,
                "type": "math",
                "correct": correct,
                "attempts": bot_config.CAPTCHA_MAX_ATTEMPTS,
                "mention": mention,
            }

        task = asyncio.create_task(self._captcha_timeout(user.id))
        self.pending_captchas[user.id]["task"] = task

    async def _captcha_timeout(self, user_id: int):
        await asyncio.sleep(bot_config.CAPTCHA_TIMEOUT_SECONDS)
        info = self.pending_captchas.get(user_id)
        if not info:
            return
        chat_id = info["chat_id"]
        try:
            await self.application.bot.ban_chat_member(chat_id, user_id)
            await self.application.bot.send_message(chat_id, f"❌ {info['mention']} failed verification and was removed.", parse_mode="HTML")
            try:
                await self.application.bot.delete_message(chat_id, info["message_id"])
            except Exception:
                pass
        finally:
            self.pending_captchas.pop(user_id, None)

    async def _verify_user(self, user_id: int, context: ContextTypes.DEFAULT_TYPE):
        info = self.pending_captchas.get(user_id)
        if not info:
            return
        task = info.get("task")
        if task:
            task.cancel()
        chat_id = info["chat_id"]
        await context.bot.restrict_chat_member(
            chat_id,
            user_id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_media_messages=True,
                can_send_polls=True,
                can_send_other_messages=True,
                can_add_web_page_previews=True,
                can_invite_users=True,
            ),
        )
        try:
            await context.bot.delete_message(chat_id, info["message_id"])
        except Exception:
            pass
        await self.db_manager.add_verified_user(user_id)
        await context.bot.send_message(chat_id, f"✅ {info['mention']} verified as human.", parse_mode="HTML")
        self.pending_captchas.pop(user_id, None)

    async def _fail_verification(self, user_id: int, context: ContextTypes.DEFAULT_TYPE):
        info = self.pending_captchas.get(user_id)
        if not info:
            return
        task = info.get("task")
        if task:
            task.cancel()
        chat_id = info["chat_id"]
        await context.bot.ban_chat_member(chat_id, user_id)
        try:
            await context.bot.delete_message(chat_id, info["message_id"])
        except Exception:
            pass
        await context.bot.send_message(chat_id, f"❌ {info['mention']} failed verification and was removed.", parse_mode="HTML")
        self.pending_captchas.pop(user_id, None)

    async def handle_new_chat_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.new_chat_members:
            return
        chat_id = update.effective_chat.id
        if self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            return
        for member in update.message.new_chat_members:
            if member.is_bot:
                continue
            await self.stats_manager.log_member_join(chat_id, member.id, member.username)
            if await self.db_manager.is_user_verified(member.id):
                continue
            try:
                await context.bot.restrict_chat_member(
                    chat_id,
                    member.id,
                    permissions=ChatPermissions(
                        can_send_messages=False,
                        can_send_media_messages=False,
                        can_send_polls=False,
                        can_send_other_messages=False,
                        can_add_web_page_previews=False,
                        can_invite_users=False,
                        can_pin_messages=False,
                        can_change_info=False,
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to restrict user {member.id}: {e}")
            await self._start_captcha(member, chat_id, context)

    async def handle_left_chat_member(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.left_chat_member:
            return
        chat_id = update.effective_chat.id
        if self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            return
        member = update.message.left_chat_member
        if member and not member.is_bot:
            await self.stats_manager.log_member_leave(chat_id, member.id)

    async def captcha_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query or not query.data:
            return
        parts = query.data.split(":")
        if len(parts) != 2:
            return
        _, target_id_str = parts
        target_id = int(target_id_str)
        user_id = query.from_user.id
        if user_id != target_id:
            await query.answer("This is not your captcha.", show_alert=True)
            return
        await query.answer()
        await self._verify_user(user_id, context)

    async def captcha_math_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query or not query.data:
            return
        parts = query.data.split(":")
        if len(parts) != 3:
            return
        _, target_id_str, answer_str = parts
        target_id = int(target_id_str)
        answer = int(answer_str)
        user_id = query.from_user.id
        if user_id != target_id:
            await query.answer("This is not your captcha.", show_alert=True)
            return
        info = self.pending_captchas.get(user_id)
        if not info:
            await query.answer()
            return
        if answer == info.get("correct"):
            await query.answer("Correct!")
            await self._verify_user(user_id, context)
        else:
            info["attempts"] -= 1
            if info["attempts"] <= 0:
                await query.answer("Wrong answer.")
                await self._fail_verification(user_id, context)
            else:
                await query.answer("Wrong, try again.")
                question, options, correct = self._generate_math_captcha()
                info["correct"] = correct
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton(str(opt), callback_data=f"captcha_math:{user_id}:{opt}") for opt in options]]
                )
                text = f"Solve: {question} = ?\nAttempts left: {info['attempts']}"
                try:
                    await query.edit_message_text(text, reply_markup=keyboard)
                except Exception:
                    pass

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

    async def handle_video_note_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.video_note:
            return

        chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ['group', 'supergroup']
        if is_group and self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            logger.debug(f"handle_video_note_message: Skipping video note from unallowed group {chat_id}.")
            return

        self.language_service._set_current_language_internals('ru')
        await update.message.reply_text(self.language_service.get_response_string("video_message_received"))

        video_file = await update.message.video_note.get_file()

        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")

        try:
            await video_file.download_to_drive(temp_file_path)

            transcribed_text = await self.llm_services.transcribe_audio(temp_file_path)

            if not transcribed_text:
                await update.message.reply_text(self.language_service.get_response_string("video_transcription_failed"))
                return

            transcription_header = self.language_service.get_response_string("video_transcription_header")
            await update.message.reply_text(f"*{transcription_header}*\n\n`{transcribed_text}`", parse_mode='MarkdownV2')

            is_russian = bool(re.search('[а-яА-Я]', transcribed_text))

            if not is_russian:
                logger.info("Transcribed text is not Russian. Proceeding with translation.")

                translation_prompts = self.language_service.get_llm_prompt_set("text_translator")

                if translation_prompts and "system" in translation_prompts and "user_template" in translation_prompts:
                    translated_text = await self.response_generator.translate_text(
                        text_to_translate=transcribed_text,
                        target_language="Russian",
                        system_prompt=translation_prompts["system"],
                        user_prompt_template=translation_prompts["user_template"]
                    )

                    if translated_text:
                        translation_header = self.language_service.get_response_string("video_translation_header")
                        await update.message.reply_text(f"*{translation_header}*\n\n`{translated_text}`", parse_mode='MarkdownV2')
                    else:
                        logger.error("Translation failed for transcribed video note.")
                else:
                    logger.error("Could not find 'text_translator' prompts in language pack.")

        except Exception as e:
            logger.error(f"Error processing video note: {e}", exc_info=True)
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary audio file: {temp_file_path}")

    async def handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ['group', 'supergroup']
        if is_group and self.allowed_group_ids and chat_id not in self.allowed_group_ids:
            logger.debug(f"handle_photo_message: Skipping photo from unallowed group {chat_id}.")
            return

        if not await self.db_manager.get_nsfw_filter_enabled(chat_id):
            return

        file_obj = None
        ext = 'jpg'
        if update.message.photo:
            photo = update.message.photo[-1]
            file_obj = await photo.get_file()
        elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith('image/'):
            file_obj = await update.message.document.get_file()
            if update.message.document.file_name and '.' in update.message.document.file_name:
                ext = update.message.document.file_name.rsplit('.', 1)[-1]
        else:
            return

        temp_dir = 'temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{ext}")
        await file_obj.download_to_drive(temp_file_path)

        try:
            if self.nsfw_classifier is None:
                if NudeClassifier is None:
                    logger.error("NSFW classifier library not installed. Install nudenet to enable filtering.")
                    return
                try:
                    self.nsfw_classifier = NudeClassifier()
                except Exception as e:
                    logger.error(f"Failed to initialize NSFW classifier: {e}", exc_info=True)
                    return

            result = self.nsfw_classifier.classify(temp_file_path)
            nsfw_score = result.get(temp_file_path, {}).get('unsafe', 0)
            if nsfw_score >= bot_config.NSFW_DETECTION_THRESHOLD:
                try:
                    await update.message.delete()
                    if update.effective_user:
                        await context.bot.send_message(
                            chat_id,
                            f"\uD83D\uDEAB Removed an NSFW image from {update.effective_user.mention_html()}.",
                            parse_mode='HTML'
                        )
                    else:
                        await context.bot.send_message(chat_id, "\uD83D\uDEAB Removed an NSFW image.")
                except Exception as e:
                    logger.error(f"Error deleting NSFW image: {e}", exc_info=True)
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

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
        user_msg_txt, triggered_by_prefix = self._extract_ai_trigger(update.message.text)
        user_msg_txt_lower = user_msg_txt.lower()

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

        draw_request = bool(triggered_by_prefix and re.match(r'^\s*draw\b', user_msg_txt_lower))
        if await self.spam_detector.inspect_message(update, context):
            return ConversationHandler.END
        if not (triggered_by_prefix or await self._is_triggered(update, context, user_msg_txt_lower)):
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
        
        if draw_request:
            master_intent = "IMAGE_GENERATION_QUERY"

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

    async def chat_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        stats = await self.stats_manager.get_chat_stats(update.effective_chat.id)
        if not stats:
            await update.message.reply_text("No statistics available yet.")
            return
        lines = [
            "Chat Statistics:",
            f"Total messages: {stats['total_messages']}",
            f"Joins: {stats['joins']} | Leaves: {stats['leaves']}"
        ]
        if stats['top_users']:
            lines.append("Top users:")
            for u in stats['top_users']:
                name = f"@{u['username']}" if u.get('username') else f"@{u['user_id']}"
                lines.append(f"- {name}: {u['count']}")
        if stats['top_links']:
            lines.append("Top links:")
            for l in stats['top_links']:
                lines.append(f"- {l['domain']}: {l['count']}")
        await update.message.reply_text("\n".join(lines))

    async def my_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat or not update.effective_user:
            return
        stats = await self.stats_manager.get_user_stats(update.effective_chat.id, update.effective_user.id)
        if not stats:
            await update.message.reply_text("No statistics available for you yet.")
            return
        total = stats['total_messages'] or 1
        percent = (stats['messages'] / total) * 100
        first_seen = stats['first_seen'] or datetime.utcnow()
        days = max((datetime.utcnow() - first_seen).days + 1, 1)
        avg_per_day = stats['messages'] / days
        lines = [
            f"You have sent {stats['messages']} messages ({percent:.1f}% of total).",
            f"First seen: {first_seen:%Y-%m-%d}",
            f"Average per day: {avg_per_day:.1f}",
            f"Rank: {stats['rank']} of {stats['total_users']}"
        ]
        await update.message.reply_text("\n".join(lines))

    async def user_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        target_user = None
        if update.message.reply_to_message and update.message.reply_to_message.from_user:
            target_user = update.message.reply_to_message.from_user
        elif context.args:
            arg = context.args[0]
            if arg.startswith('@'):
                username = arg[1:].lower()
                stats = self.stats_manager.memory_stats.get(update.effective_chat.id, {})
                for uid, info in stats.get('users', {}).items():
                    if info.get('username') and info['username'].lower() == username:
                        target_user = SimpleNamespace(id=uid, username=info['username'], full_name=info['username'])
                        break
            elif arg.isdigit():
                try:
                    member = await context.bot.get_chat_member(update.effective_chat.id, int(arg))
                    target_user = member.user
                except Exception:
                    target_user = SimpleNamespace(id=int(arg), username=None, full_name=arg)
        if not target_user:
            await update.message.reply_text("Reply to a user's message or provide a user ID/username.")
            return
        stats = await self.stats_manager.get_user_stats(update.effective_chat.id, target_user.id)
        if not stats:
            await update.message.reply_text("No statistics available for this user yet.")
            return
        total = stats['total_messages'] or 1
        percent = (stats['messages'] / total) * 100
        first_seen = stats['first_seen'] or datetime.utcnow()
        days = max((datetime.utcnow() - first_seen).days + 1, 1)
        avg_per_day = stats['messages'] / days
        name = target_user.full_name if getattr(target_user, 'full_name', None) else (target_user.username or str(target_user.id))
        lines = [
            f"Stats for {name}:",
            f"Messages: {stats['messages']} ({percent:.1f}% of total)",
            f"First seen: {first_seen:%Y-%m-%d}",
            f"Average per day: {avg_per_day:.1f}",
            f"Rank: {stats['rank']} of {stats['total_users']}"
        ]
        await update.message.reply_text("\n".join(lines))

    async def news_command_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        if not update.message or not update.effective_user: return ConversationHandler.END
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE)
        context.user_data['conversation_state'] = ASK_NEWS_TOPIC
        return await self.news_handler.handle_command_entry(update, context)

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a message with /report to alert admins.")
            return
        reported = update.message.reply_to_message
        chat_id = update.effective_chat.id
        report_text = f"\U0001F6A8 Reported message by @{reported.from_user.username or reported.from_user.id}:"  # 🚨
        if reported.text:
            report_text += f"\n{reported.text}"
        try:
            if bot_config.REPORTS_CHANNEL_ID:
                await context.bot.forward_message(bot_config.REPORTS_CHANNEL_ID, chat_id, reported.message_id)
                await context.bot.send_message(bot_config.REPORTS_CHANNEL_ID, report_text)
            else:
                admins = await context.bot.get_chat_administrators(chat_id)
                for admin in admins:
                    if not admin.user.is_bot:
                        try:
                            await context.bot.send_message(admin.user.id, report_text)
                        except Exception as e:
                            logger.error(f"Failed to notify admin {admin.user.id}: {e}")
            await update.message.reply_text("Thanks, the admins have been notified.")
        except Exception as e:
            logger.error(f"Error handling report command: {e}", exc_info=True)

    async def spam_vote_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a message with /spam to vote for banning the sender.")
            return
        chat_id = update.effective_chat.id
        reporter_id = update.effective_user.id if update.effective_user else None
        if reporter_id is None:
            return
        try:
            member = await context.bot.get_chat_member(chat_id, reporter_id)
            if member.status in ("administrator", "creator"):
                await update.message.reply_text("Admins can ban users directly without voting.")
                return
        except Exception:
            pass
        target_user = update.message.reply_to_message.from_user
        try:
            target_member = await context.bot.get_chat_member(chat_id, target_user.id)
            if target_member.status in ("administrator", "creator"):
                await update.message.reply_text("Cannot vote to ban an admin.")
                return
        except Exception:
            pass
        added = await self.db_manager.add_spam_vote(chat_id, target_user.id, reporter_id)
        if not added:
            await update.message.reply_text("You have already voted to ban this user.")
            return
        threshold = await self.db_manager.get_spam_vote_threshold(chat_id, bot_config.DEFAULT_SPAM_VOTE_THRESHOLD)
        count = await self.db_manager.count_spam_votes(chat_id, target_user.id, bot_config.SPAM_VOTE_TIME_WINDOW_MINUTES)
        if count >= threshold:
            try:
                await context.bot.ban_chat_member(chat_id, target_user.id)
                await update.message.reply_html(
                    f"User {target_user.mention_html()} was banned by community vote for spam."
                )
                try:
                    await context.bot.delete_message(chat_id, update.message.reply_to_message.message_id)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to ban user {target_user.id}: {e}")
            await self.db_manager.clear_spam_votes(chat_id, target_user.id)
        else:
            await update.message.reply_text(f"Spam vote recorded ({count}/{threshold}).")

    async def ban_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to ban them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        try:
            await context.bot.ban_chat_member(chat_id, target.id)
            await update.message.reply_html(
                f"User {target.mention_html()} has been banned by {update.effective_user.mention_html()}.")
        except Exception as e:
            logger.error(f"Failed to ban user {target.id}: {e}")
            await update.message.reply_text("Failed to ban user.")

    async def unban_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to unban them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        try:
            await context.bot.unban_chat_member(chat_id, target.id)
            await update.message.reply_html(
                f"User {target.mention_html()} has been unbanned by {update.effective_user.mention_html()}.")
        except Exception as e:
            logger.error(f"Failed to unban user {target.id}: {e}")
            await update.message.reply_text("Failed to unban user.")

    async def kick_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to kick them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        try:
            await context.bot.ban_chat_member(chat_id, target.id)
            await context.bot.unban_chat_member(chat_id, target.id)
            await update.message.reply_html(
                f"User {target.mention_html()} has been kicked by {update.effective_user.mention_html()}.")
        except Exception as e:
            logger.error(f"Failed to kick user {target.id}: {e}")
            await update.message.reply_text("Failed to kick user.")

    async def mute_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to mute them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        duration = 10
        if context.args and context.args[0].isdigit():
            duration = int(context.args[0])
        until_date = datetime.utcnow() + timedelta(minutes=duration)
        perms = ChatPermissions(can_send_messages=False, can_send_media_messages=False,
                                can_send_polls=False, can_send_other_messages=False,
                                can_add_web_page_previews=False)
        try:
            await context.bot.restrict_chat_member(chat_id, target.id, permissions=perms, until_date=until_date)
            await update.message.reply_html(
                f"User {target.mention_html()} has been muted for {duration} minutes.")
        except Exception as e:
            logger.error(f"Failed to mute user {target.id}: {e}")
            await update.message.reply_text("Failed to mute user.")

    async def unmute_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to unmute them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        perms = ChatPermissions(can_send_messages=True, can_send_media_messages=True,
                                can_send_polls=True, can_send_other_messages=True,
                                can_add_web_page_previews=True)
        try:
            await context.bot.restrict_chat_member(chat_id, target.id, permissions=perms)
            await update.message.reply_html(
                f"User {target.mention_html()} has been unmuted.")
        except Exception as e:
            logger.error(f"Failed to unmute user {target.id}: {e}")
            await update.message.reply_text("Failed to unmute user.")

    async def warn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to warn them.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        reason = " ".join(context.args) if context.args else None
        count = await self.db_manager.add_warning(chat_id, target.id, reason)
        await update.message.reply_html(
            f"\u26A0\uFE0F {target.mention_html()} has been warned (Warn #{count}).")
        if count >= 3:
            try:
                await context.bot.ban_chat_member(chat_id, target.id)
                await update.message.reply_html(
                    f"User {target.mention_html()} was banned after reaching 3 warnings.")
            except Exception as e:
                logger.error(f"Auto-ban after warnings failed for user {target.id}: {e}")

    async def warns_list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        if update.message.reply_to_message:
            target = update.message.reply_to_message.from_user
            count = await self.db_manager.get_warning_count(chat_id, target.id)
            await update.message.reply_text(f"User {target.id} has {count} warning(s).")
            return
        rows = await self.db_manager.list_warnings(chat_id)
        if not rows:
            await update.message.reply_text("No warnings recorded.")
            return
        lines = [f"{user_id}: {warns}" for user_id, warns in rows]
        await update.message.reply_text("Warnings:\n" + "\n".join(lines))

    async def remove_warn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to clear their warnings.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        await self.db_manager.clear_warnings(chat_id, target.id)
        await update.message.reply_text(f"Warnings cleared for user {target.id}.")

    async def toggle_nsfw_filter_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        if not await self._is_user_admin(chat_id, user_id, context):
            return
        if not context.args or context.args[0].lower() not in ("on", "off"):
            await update.message.reply_text("Usage: /toggle_nsfw <on|off>")
            return
        enabled = context.args[0].lower() == "on"
        await self.db_manager.set_nsfw_filter_enabled(chat_id, enabled)
        status = "enabled" if enabled else "disabled"
        await update.message.reply_text(f"NSFW filter {status} for this chat.")

    async def set_spam_threshold_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        try:
            member = await context.bot.get_chat_member(chat_id, user_id)
            if member.status not in ("administrator", "creator"):
                return
        except Exception:
            return
        if not context.args or not context.args[0].isdigit():
            await update.message.reply_text("Usage: /setspamthreshold <number>")
            return
        threshold = int(context.args[0])
        await self.db_manager.set_spam_vote_threshold(chat_id, threshold)
        await update.message.reply_text(f"Spam vote threshold set to {threshold}.")

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
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler(["spam", "voteban"], self.spam_vote_command))
        self.application.add_handler(CommandHandler("ban", self.ban_command))
        self.application.add_handler(CommandHandler(["unban", "pardon"], self.unban_command))
        self.application.add_handler(CommandHandler("kick", self.kick_command))
        self.application.add_handler(CommandHandler("mute", self.mute_command))
        self.application.add_handler(CommandHandler("unmute", self.unmute_command))
        self.application.add_handler(CommandHandler(["stat", "stats"], self.chat_stats_command))
        self.application.add_handler(CommandHandler("mystat", self.my_stats_command))
        self.application.add_handler(CommandHandler("userstats", self.user_stats_command))
        self.application.add_handler(CommandHandler("warn", self.warn_command))
        self.application.add_handler(CommandHandler("warns_list", self.warns_list_command))
        self.application.add_handler(CommandHandler(["rm_warn", "clear_warn"], self.remove_warn_command))
        self.application.add_handler(CommandHandler("toggle_nsfw", self.toggle_nsfw_filter_command))
        self.application.add_handler(CommandHandler("setspamthreshold", self.set_spam_threshold_command))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, self.handle_new_chat_members))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.LEFT_CHAT_MEMBER, self.handle_left_chat_member))
        self.application.add_handler(CallbackQueryHandler(self.captcha_button_callback, pattern=r"^captcha_button:"))
        self.application.add_handler(CallbackQueryHandler(self.captcha_math_callback, pattern=r"^captcha_math:"))
        
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

        # Add the standalone handlers for voice and video note messages
        self.application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.handle_photo_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
        self.application.add_handler(MessageHandler(filters.VIDEO_NOTE, self.handle_video_note_message))
        
        self.application.add_error_handler(self.error_handler)
        logger.info("TelegramHandlerService: All handlers registered.")
