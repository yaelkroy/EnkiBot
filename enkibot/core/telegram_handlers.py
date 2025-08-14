# enkibot/core/telegram_handlers.py
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

from telegram import (
    Update,
    ReplyKeyboardRemove,
    ChatPermissions,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeChat,
    BotCommandScopeChatAdministrators,
)
from telegram.ext import (
    Application,
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
# Optional import for message reaction updates
try:  # pragma: no cover - optional dependency
    from telegram.ext import MessageReactionHandler
except Exception:  # pragma: no cover - fallback when not supported
    MessageReactionHandler = None
from telegram.constants import ChatAction
from telegram.helpers import mention_html
import re
import random
from datetime import datetime, timedelta

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
    from enkibot.modules.karma_manager import KarmaManager
    from enkibot.modules.community_moderation import CommunityModerationService
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
from enkibot.utils.message_utils import is_forwarded_message, clean_output_text
from enkibot.utils.trigger_extractor import extract_assistant_prompt
from enkibot.utils.text_splitter import split_text_into_chunks

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
                 karma_manager: 'KarmaManager',
                 community_moderation: 'CommunityModerationService',
                 allowed_group_ids: set,
                 bot_nicknames: list,
                 general_handler: 'GeneralIntentHandler' # Add new handler instance here
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
        self.karma_manager = karma_manager
        self.community_moderation = community_moderation
        
        self.allowed_group_ids = allowed_group_ids 
        self.bot_nicknames = bot_nicknames

        self.pending_action_data: Dict[int, Dict[str, Any]] = {}
        self.pending_captchas: Dict[int, Dict[str, Any]] = {}
        self.nsfw_classifier: Optional['NudeClassifier'] = None
        # Feature flags per chat for dynamic command hints
        self.features_db: Dict[int, Dict[str, bool]] = {}
        # Manual language overrides per chat
        self.chat_languages: Dict[int, str] = {}
        # Registry of default commands for help text and Telegram registration
        self.default_commands: Dict[str, Dict[str, str]] = {}

        # Map reaction emoji to handler coroutines
        # Note: these handlers are defined below in this class. Keep this mapping here after definitions are ensured.
        self.reaction_handlers: Dict[str, Any] = {}

        # Track recent reactions to bot messages for follow-up triggers
        self.recent_reactors: Dict[int, datetime] = {}

        # Map inline refinement actions to handlers
        self.refinement_handlers: Dict[str, Any] = {}

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
        # Use the passed-in instance instead of creating a new one
        self.general_handler = general_handler
        
        self.image_generation_handler = ImageGenerationIntentHandler(
            language_service=self.language_service,
            intent_recognizer=self.intent_recognizer,
            llm_services=self.llm_services,
            db_manager=self.db_manager
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
        # Ensure chat activity is visible in the terminal even if logging is redirected
        print(
            f"CHAT[{chat_id}] {user.username or user.id}: {message.text}"
            f" | profile action: {action_taken}"
        )
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
        if not update.message or not context.bot:
            return False

        current_chat_id = update.effective_chat.id
        is_group = update.message.chat.type in ["group", "supergroup"]
        if not is_group:
            return True

        if self.allowed_group_ids and current_chat_id not in self.allowed_group_ids:
            return False

        if is_forwarded_message(update.message):
            logger.info(f"_is_triggered: True (Forwarded message in chat {current_chat_id})")
            return True

        bot_username_lower = getattr(context.bot, "username", "").lower() if getattr(context.bot, "username", None) else ""
        is_at_mentioned = bool(bot_username_lower and f"@{bot_username_lower}" in user_msg_txt_lower)
        tokens = re.findall(r"\w+", user_msg_txt_lower, flags=re.UNICODE)
        is_nickname_mentioned = any(nick.lower() in tokens for nick in self.bot_nicknames)
        is_bot_mentioned = is_at_mentioned or is_nickname_mentioned
        is_reply_to_bot = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and context.bot
            and update.message.reply_to_message.from_user.id == context.bot.id
        )
        final_trigger_decision = is_bot_mentioned or is_reply_to_bot
        if is_group and final_trigger_decision:
            logger.info(
                f"_is_triggered: True (Group: {current_chat_id}): @M={is_at_mentioned}, NickM={is_nickname_mentioned}, Reply={is_reply_to_bot}"
            )
        return final_trigger_decision

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

    async def start_captcha(self, user, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
        """Public wrapper to initiate captcha verification for a user.

        This exposes the existing captcha mechanism to other modules
        (e.g., the spam detector) without requiring them to know about the
        private ``_start_captcha`` implementation details.
        """
        await self._start_captcha(user, chat_id, context)

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
                can_send_audios=True,
                can_send_documents=True,
                can_send_photos=True,
                can_send_videos=True,
                can_send_video_notes=True,
                can_send_voice_notes=True,
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
            # Print to console for visibility of join events
            print(
                f"JOIN[{chat_id}] {member.username or member.id} joined the chat"
            )
            try:
                await context.bot.restrict_chat_member(
                    chat_id,
                    member.id,
                    permissions=ChatPermissions(
                        can_send_messages=False,
                        can_send_audios=False,
                        can_send_documents=False,
                        can_send_photos=False,
                        can_send_videos=False,
                        can_send_video_notes=False,
                        can_send_voice_notes=False,
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
            # Print to console for visibility of leave events
            print(
                f"LEAVE[{chat_id}] {member.username or member.id} left the chat"
            )

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

    async def refinement_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if not query or not query.data:
            return
        parts = query.data.split(":")
        if len(parts) != 2:
            await query.answer()
            return
        _, action = parts
        handler = self.refinement_handlers.get(action)
        await query.answer()
        if handler:
            await handler(update, context)

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
            threshold = await self.db_manager.get_nsfw_threshold(chat_id)
            if nsfw_score >= threshold:
                try:
                    await update.message.delete()
                    if update.effective_user:
                        text = self.language_service.get_response_string(
                            "nsfw_image_removed_user",
                            user=update.effective_user.mention_html(),
                        )
                        await context.bot.send_message(
                            chat_id,
                            text,
                            parse_mode='HTML',
                        )
                    else:
                        text = self.language_service.get_response_string(
                            "nsfw_image_removed_chat"
                        )
                        await context.bot.send_message(chat_id, text)
                except Exception as e:
                    logger.error(f"Error deleting NSFW image: {e}", exc_info=True)
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    async def _handle_karma_vote(self, update: Update) -> bool:
        """Process simple text-based karma votes."""
        message = update.message
        if not message or not message.reply_to_message or not message.text:
            return False
        giver = update.effective_user
        receiver = message.reply_to_message.from_user if message.reply_to_message else None
        if not giver or not receiver:
            return False
        vote_result = await self.karma_manager.handle_text_vote(
            giver.id, receiver.id, message.chat_id, message.text
        )
        if vote_result is None:
            return False
        result, delta = vote_result
        receiver_display = f"@{receiver.username}" if receiver.username else receiver.first_name
        if result == "self_karma_error":
            await message.reply_text(
                self.language_service.get_response_string(
                    "karma_self_vote_error", "🤷 You can't vote on yourself."
                )
            )
        elif result == "cooldown_error":
            await message.reply_text(
                self.language_service.get_response_string(
                    "karma_vote_cooldown",
                    "⏱ You can vote for this user again later.",
                )
            )
        elif result == "karma_changed_success":
            stats = await self.karma_manager.get_user_stats(receiver.id)
            total = stats["received"] if stats else 0
            sign = "+" if delta > 0 else ""
            await message.reply_text(
                self.language_service.get_response_string(
                    "karma_vote_success",
                    f"{sign}{delta} to {receiver_display} (total {total:+})",
                    sign=sign,
                    delta=delta,
                    receiver=receiver_display,
                    total=total,
                )
            )
        return True

    async def _handle_regenerate_reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regeneration request via emoji reaction."""
        reactor = getattr(getattr(update, "message_reaction", None), "user", None)
        lang_code = getattr(reactor, "language_code", None)
        if lang_code and lang_code in self.language_service.language_packs:
            self.language_service._set_current_language_internals(lang_code)
        else:
            self.language_service._set_current_language_internals(self.language_service.default_language)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.language_service.get_response_string(
                "reaction_regenerating", "Regenerating response..."
            ),
        )

    async def _handle_expand_reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle expansion request via emoji reaction."""
        reactor = getattr(getattr(update, "message_reaction", None), "user", None)
        lang_code = getattr(reactor, "language_code", None)
        if lang_code and lang_code in self.language_service.language_packs:
            self.language_service._set_current_language_internals(lang_code)
        else:
            self.language_service._set_current_language_internals(self.language_service.default_language)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.language_service.get_response_string(
                "reaction_expanding", "Expanding on previous response..."
            ),
        )

    async def _handle_summary_reaction(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle summarization request via emoji reaction."""
        reactor = getattr(getattr(update, "message_reaction", None), "user", None)
        lang_code = getattr(reactor, "language_code", None)
        if lang_code and lang_code in self.language_service.language_packs:
            self.language_service._set_current_language_internals(lang_code)
        else:
            self.language_service._set_current_language_internals(self.language_service.default_language)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.language_service.get_response_string(
                "reaction_summary", "Generating summary..."
            ),
        )

    async def reaction_router(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Route emoji reactions to configured actions."""
        reaction_update = getattr(update, "message_reaction", None)
        if not reaction_update or not reaction_update.new_reaction:
            return
        emoji = reaction_update.new_reaction[0].emoji
        try:
            msg = getattr(reaction_update, "message", None)
            rater = getattr(reaction_update, "user", None)
            if msg and msg.from_user and rater:
                if context.bot and msg.from_user.id == context.bot.id:
                    self.recent_reactors[rater.id] = datetime.utcnow()
                await self.karma_manager.record_reaction_event(
                    chat_id=update.effective_chat.id,
                    msg_id=msg.message_id,
                    target_user_id=msg.from_user.id,
                    rater_user_id=rater.id,
                    emoji=emoji,
                )
        except Exception as exc:
            logger.error(f"Karma reaction logging failed: {exc}")
        handler = self.reaction_handlers.get(emoji)
        if handler:
            await handler(update, context)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        if not update.message or not update.message.text or not update.effective_chat or not update.effective_user:
            return None

        chat_id = update.effective_chat.id
        if chat_id in self.chat_languages:
            self.language_service._set_current_language_internals(self.chat_languages[chat_id])
        else:
            await self.language_service.determine_language_context(
                update.message.text,
                chat_id=chat_id,
                update_context=update
            )

        await self.log_message_and_profile_tasks(update, context)

        if await self._handle_karma_vote(update):
            return None

        text = update.message.text
        bot_username_lower = getattr(context.bot, 'username', "").lower() if getattr(context.bot, 'username', None) else ""
        user_id = update.effective_user.id
        is_reply_to_bot = (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and context.bot
            and update.message.reply_to_message.from_user.id == context.bot.id
        )
        reason = ""
        alias = ""
        now = datetime.utcnow()
        recent_react = self.recent_reactors.get(user_id)
        if recent_react and now - recent_react < timedelta(seconds=30):
            triggered = True
            user_msg_txt = text.strip()
            reason = "reaction"
            self.recent_reactors.pop(user_id, None)
        elif is_reply_to_bot:
            triggered = True
            user_msg_txt = text.strip()
            reason = "reply"
        else:
            triggered, user_msg_txt, alias = extract_assistant_prompt(text, self.bot_nicknames, bot_username_lower)
            reason = "name" if triggered else ""
            if not triggered:
                triggered = await self._is_triggered(update, context, text.lower())
                if triggered:
                    reason = "forward" if is_forwarded_message(update.message) else "mention"
                    user_msg_txt = text.strip()

        if not triggered:
            return None

        if not user_msg_txt:
            nudge = self.language_service.get_response_string("assistant_prompt_nudge", "Я здесь. О чём рассказать? 🙂")
            await update.message.reply_text(nudge)
            await self.db_manager.log_assistant_invocation(chat_id, user_id, update.message.message_id, True, alias, user_msg_txt, reason, self.language_service.current_lang, False, False, None)
            return None

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

        draw_request = bool(re.match(r'^\s*draw\b', user_msg_txt_lower))
        if await self.spam_detector.inspect_message(update, context):
            return ConversationHandler.END

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
        # Mirror key intent decisions to the console
        print(
            f"ACTION[{chat_id}] intent={master_intent} reason={reason}"
        )

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

        await self.db_manager.log_assistant_invocation(
            chat_id,
            user_id,
            update.message.message_id,
            True,
            alias,
            user_msg_txt,
            reason,
            self.language_service.current_lang,
            True,
            True,
            None,
        )

        if next_state is None or next_state == ConversationHandler.END:
            if context.user_data and 'conversation_state' in context.user_data:
                context.user_data.pop('conversation_state')
            if chat_id in self.pending_action_data:
                 self.pending_action_data.pop(chat_id, None)
        return next_state

    async def _handle_message_analysis_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> None:
        is_valid_reply_scenario = (
            update.message and update.message.reply_to_message and
            (update.message.reply_to_message.text or update.message.reply_to_message.caption) and
            update.message.reply_to_message.from_user and context.bot and
            update.message.reply_to_message.from_user.id != context.bot.id )
        if not is_valid_reply_scenario:
            logger.info("MESSAGE_ANALYSIS_QUERY classified, but not valid reply. Delegating to GeneralIntentHandler.")
            await self.general_handler.handle_request(update, context, user_msg_txt, "GENERAL_CHAT")
            return
        
        logger.info("TelegramHandlers: Processing MESSAGE_ANALYSIS_QUERY directly.")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        original_msg = update.message.reply_to_message
        original_text = original_msg.text or original_msg.caption or ""
        question_for_analysis = user_msg_txt
        bot_username_lower = getattr(context.bot, 'username', "").lower() if getattr(context.bot, 'username', None) else ""
        cleaned_question = user_msg_txt.lower()
        for nick in self.bot_nicknames + ([f"@{bot_username_lower}"] if bot_username_lower else []): 
            cleaned_question = cleaned_question.replace(nick.lower(), "").strip()
        if len(cleaned_question) < 5: 
            question_for_analysis = self.language_service.get_response_string("replied_message_default_question")
        
        if is_forwarded_message(original_msg):
            # The direct fact-check is now handled in general_handler.py, so we don't need this block.
            # We can still route it to the general handler for generic analysis.
            await self.general_handler.handle_request(update, context, user_msg_txt, "GENERAL_CHAT")
            return
        else:
            analyzer_prompts = self.language_service.get_llm_prompt_set("replied_message_analyzer")
            if not (analyzer_prompts and "system" in analyzer_prompts):
                logger.error("Prompt set for replied message analysis is missing or malformed.")
                await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
                return
            analysis_result = await self.response_generator.analyze_replied_message(
                original_text=original_text,
                user_question=question_for_analysis,
                system_prompt=analyzer_prompts["system"],
                user_prompt_template=analyzer_prompts.get("user_template")
            )
        await update.message.reply_text(analysis_result)
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message: return
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE) 
        await update.message.reply_html(self.language_service.get_response_string("start", user_mention=update.effective_user.mention_html()))

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send help text to the user regardless of message type."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE)
        intro = self.language_service.get_response_string("help")
        commands_text = "\n".join(
            f"/{name} - {info['long']}" for name, info in self.default_commands.items()
        )
        text = f"{intro}\n\n**Commands:**\n{commands_text}"
        if update.message:
            await update.message.reply_text(text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)

    async def chat_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return
        stats = await self.stats_manager.get_chat_stats(chat_id)
        if not stats:
            if update.message:
                await update.message.reply_text("No statistics available yet.")
            else:
                await context.bot.send_message(chat_id=chat_id, text="No statistics available yet.")
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
                lines.append(f"- {mention_html(u['user_id'], name)}: {u['count']}")
        if stats['top_links']:
            lines.append("Top links:")
            for l in stats['top_links']:
                lines.append(f"- {l['domain']}: {l['count']}")
        output = "\n".join(lines)
        if update.message:
            await update.message.reply_html(output)
        else:
            await context.bot.send_message(chat_id=chat_id, text=output, parse_mode='HTML')

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

    async def karma_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return
        top_users = await self.karma_manager.get_top_users(chat_id)
        if not top_users and update.effective_chat.type == "private":
            # Fallback to global top in private chats
            top_users = await self.karma_manager.get_global_top()
        if not top_users:
            if update.message:
                await update.message.reply_text("No karma data available.")
            else:
                await context.bot.send_message(chat_id=chat_id, text="No karma data available.")
            return
        lines = ["Karma leaderboard:"]
        for idx, u in enumerate(top_users, start=1):
            lines.append(f"{idx}. {u['name']}: {u['score']}")
        output = "\n".join(lines)
        if update.message:
            await update.message.reply_text(output)
        else:
            await context.bot.send_message(chat_id=chat_id, text=output)

    async def language_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        if not context.args:
            await update.message.reply_text("Usage: /language <code>")
            return
        lang_code = context.args[0].lower()
        if lang_code not in self.language_service.language_packs:
            await update.message.reply_text("Unsupported language code.")
            return
        self.chat_languages[chat_id] = lang_code
        self.language_service._set_current_language_internals(lang_code)
        await update.message.reply_text(f"Language for this chat set to {lang_code}.")

    async def reload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        self.chat_languages.pop(chat_id, None)
        self.stats_manager.memory_stats.pop(chat_id, None)
        self.features_db.pop(chat_id, None)
        await self.refresh_chat_commands(chat_id)
        await update.message.reply_text("Chat data reloaded.")

    async def news_command_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        if not update.message or not update.effective_user: return ConversationHandler.END
        self.language_service._set_current_language_internals(bot_config.DEFAULT_LANGUAGE)
        context.user_data['conversation_state'] = ASK_NEWS_TOPIC
        return await self.news_handler.handle_command_entry(update, context)

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.community_moderation.cmd_report(update, context)

    async def spam_vote_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.community_moderation.cmd_vote(update, context, reason="spam")

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

    async def qban_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        except Exception as e:
            logger.error(f"Failed to ban user {target.id}: {e}")
            await update.message.reply_text("Failed to ban user.")
            return
        try:
            await update.message.reply_to_message.delete()
        except Exception:
            pass
        try:
            await update.message.delete()
        except Exception:
            pass

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

    async def baninfo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to get ban info.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        try:
            member = await context.bot.get_chat_member(chat_id, target.id)
            if member.status == "kicked":
                if member.until_date:
                    await update.message.reply_html(
                        f"User {target.mention_html()} is banned until {member.until_date}.")
                else:
                    await update.message.reply_html(
                        f"User {target.mention_html()} is banned permanently.")
            else:
                await update.message.reply_text("User is not banned.")
        except Exception as e:
            logger.error(f"Failed to get ban info for user {target.id}: {e}")
            await update.message.reply_text("Failed to get ban info.")

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
        perms = ChatPermissions(
            can_send_messages=False,
            can_send_audios=False,
            can_send_documents=False,
            can_send_photos=False,
            can_send_videos=False,
            can_send_video_notes=False,
            can_send_voice_notes=False,
            can_send_polls=False,
            can_send_other_messages=False,
            can_add_web_page_previews=False,
        )
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
        perms = ChatPermissions(
            can_send_messages=True,
            can_send_audios=True,
            can_send_documents=True,
            can_send_photos=True,
            can_send_videos=True,
            can_send_video_notes=True,
            can_send_voice_notes=True,
            can_send_polls=True,
            can_send_other_messages=True,
            can_add_web_page_previews=True,
        )
        try:
            await context.bot.restrict_chat_member(chat_id, target.id, permissions=perms)
            await update.message.reply_html(
                f"User {target.mention_html()} has been unmuted.")
        except Exception as e:
            logger.error(f"Failed to unmute user {target.id}: {e}")
            await update.message.reply_text("Failed to unmute user.")

    async def muteinfo_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.reply_to_message:
            await update.message.reply_text("Reply to a user's message to get mute info.")
            return
        chat_id = update.effective_chat.id
        invoker_id = update.effective_user.id if update.effective_user else 0
        if not await self._is_user_admin(chat_id, invoker_id, context):
            return
        target = update.message.reply_to_message.from_user
        try:
            member = await context.bot.get_chat_member(chat_id, target.id)
            can_send = getattr(member, 'can_send_messages', True)
            if member.status == "restricted" or not can_send:
                if member.until_date:
                    await update.message.reply_html(
                        f"User {target.mention_html()} is muted until {member.until_date}.")
                else:
                    await update.message.reply_html(
                        f"User {target.mention_html()} is muted indefinitely.")
            else:
                await update.message.reply_text("User is not muted.")
        except Exception as e:
            logger.error(f"Failed to get mute info for user {target.id}: {e}")
            await update.message.reply_text("Failed to get mute info.")

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
            await update.message.reply_text(
                self.language_service.get_response_string("toggle_nsfw_usage")
            )
            return
        enabled = context.args[0].lower() == "on"
        await self.db_manager.set_nsfw_filter_enabled(chat_id, enabled)
        status_key = "enabled" if enabled else "disabled"
        status_txt = self.language_service.get_response_string(f"status_{status_key}")
        await update.message.reply_text(
            self.language_service.get_response_string(
                "nsfw_filter_status", status=status_txt
            )
        )

    async def set_nsfw_threshold_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        if not await self._is_user_admin(chat_id, user_id, context):
            return
        if not context.args:
            await update.message.reply_text(
                self.language_service.get_response_string("nsfw_threshold_usage")
            )
            return
        try:
            threshold = float(context.args[0])
        except ValueError:
            await update.message.reply_text(
                self.language_service.get_response_string("nsfw_threshold_must_number")
            )
            return
        threshold = max(0.0, min(1.0, threshold))
        await self.db_manager.set_nsfw_threshold(chat_id, threshold)
        await update.message.reply_text(
            self.language_service.get_response_string(
                "nsfw_threshold_set", threshold=threshold
            )
        )

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
            await update.message.reply_text(
                self.language_service.get_response_string("set_spam_threshold_usage")
            )
            return
        threshold = int(context.args[0])
        await self.db_manager.set_spam_vote_threshold(chat_id, threshold)
        await update.message.reply_text(
            self.language_service.get_response_string(
                "spam_threshold_set", threshold=threshold
            )
        )

    async def push_default_commands(self) -> None:
        """Registers global default slash commands."""
        commands = [
            BotCommand(name, info["short"]) for name, info in self.default_commands.items()
        ]
        await self.application.bot.set_my_commands(commands, scope=BotCommandScopeDefault())

    async def refresh_chat_commands(self, chat_id: int) -> None:
        """Refreshes command list for a specific chat based on feature flags."""
        cfg = self.features_db.get(chat_id, {})
        cmds = [
            BotCommand("stat", "Show chat statistics"),
            BotCommand("report", "Report a message"),
            BotCommand("karma", "Show karma leaderboard"),
        ]
        if cfg.get("ai"):
            cmds.append(BotCommand("ask", "Ask AI a question"))
            cmds.append(BotCommand("draw", "Generate an image with AI"))
        if cfg.get("captcha"):
            cmds.append(BotCommand("captcha", "Show captcha settings"))
        await self.application.bot.set_my_commands(cmds, scope=BotCommandScopeChat(chat_id))
        admin_cmds = [
            BotCommand("ban", "Ban the replied user"),
            BotCommand("qban", "Quick ban and delete"),
            BotCommand("unban", "Unban the replied user"),
            BotCommand("baninfo", "Ban info of the replied user"),
            BotCommand("kick", "Kick the replied user"),
            BotCommand("mute", "Mute the replied user"),
            BotCommand("unmute", "Unmute the replied user"),
            BotCommand("muteinfo", "Mute info of the replied user"),
            BotCommand("warn", "Warn the replied user"),
            BotCommand("rm_warn", "Remove user warnings"),
            BotCommand("warns_list", "List user warnings"),
            BotCommand("language", "Set chat language"),
            BotCommand("reload", "Reload chat data"),
        ]
        await self.application.bot.set_my_commands(admin_cmds, scope=BotCommandScopeChatAdministrators(chat_id))

    async def toggle_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Admin command to toggle AI features and refresh commands."""
        if not update.message or not update.effective_user:
            return
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        if not await self._is_user_admin(chat_id, user_id, context):
            return
        self.features_db.setdefault(chat_id, {})
        self.features_db[chat_id]["ai"] = not self.features_db[chat_id].get("ai", False)
        await self.refresh_chat_commands(chat_id)
        status = "ON" if self.features_db[chat_id]["ai"] else "OFF"
        await update.message.reply_text(f"AI feature is now {status}.")

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

    def _register_command(self, names, handler, short_desc: str, long_desc: str) -> None:
        """Register command handler and store its descriptions."""
        primary = names[0] if isinstance(names, (list, tuple, set)) else names
        if handler is not None:
            self.application.add_handler(CommandHandler(names, handler))
        self.default_commands[primary] = {"short": short_desc, "long": long_desc}

    def bind_reaction_handlers(self) -> None:
        """Bind emoji/inline actions to refinement handlers. Call during handler registration."""
        self.reaction_handlers = {
            "🔄": self._handle_regenerate_reaction,
            "➕": self._handle_expand_reaction,
            "📝": self._handle_summary_reaction,
        }
        self.refinement_handlers = {
            "regenerate": self._handle_regenerate_reaction,
            "expand": self._handle_expand_reaction,
            "summary": self._handle_summary_reaction,
        }

    def register_all_handlers(self):
        self._register_command("start", self.start_command, "Start the bot", "Start interaction")
        self._register_command("help", self.help_command, "How to use the bot", "This help message")
        self._register_command("report", self.report_command, "Report a message", "Report a message")
        self._register_command(["spam", "voteban"], self.spam_vote_command, "Vote to ban a spammer", "Vote to ban a spammer")
        self.application.add_handler(CommandHandler("ban", self.ban_command))
        self.application.add_handler(CommandHandler("qban", self.qban_command))
        self.application.add_handler(CommandHandler(["unban", "pardon"], self.unban_command))
        self.application.add_handler(CommandHandler("baninfo", self.baninfo_command))
        self.application.add_handler(CommandHandler("kick", self.kick_command))
        self.application.add_handler(CommandHandler("mute", self.mute_command))
        self.application.add_handler(CommandHandler("unmute", self.unmute_command))
        self.application.add_handler(CommandHandler("muteinfo", self.muteinfo_command))
        self._register_command(["stat", "stats"], self.chat_stats_command, "Chat statistics", "Show chat statistics")
        self._register_command("mystat", self.my_stats_command, "Your statistics", "Show your statistics")
        self.application.add_handler(CommandHandler("userstats", self.user_stats_command))
        self._register_command("karma", self.karma_command, "Show karma leaderboard", "Show karma leaderboard")
        self.application.add_handler(CommandHandler("language", self.language_command))
        self.application.add_handler(CommandHandler("reload", self.reload_command))
        self.application.add_handler(CommandHandler("warn", self.warn_command))
        self.application.add_handler(CommandHandler("warns_list", self.warns_list_command))
        self.application.add_handler(CommandHandler(["rm_warn", "clear_warn"], self.remove_warn_command))
        self.application.add_handler(CommandHandler("toggle_nsfw", self.toggle_nsfw_filter_command))
        self.application.add_handler(CommandHandler("nsfw_threshold", self.set_nsfw_threshold_command))
        self.application.add_handler(CommandHandler("setspamthreshold", self.set_spam_threshold_command))
        self.application.add_handler(CommandHandler("toggle_ai", self.toggle_ai_command))
        self.application.add_handler(CommandHandler("karmadiag", self.karmadiag_command))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, self.handle_new_chat_members))
        self.application.add_handler(MessageHandler(filters.StatusUpdate.LEFT_CHAT_MEMBER, self.handle_left_chat_member))
        self.application.add_handler(CallbackQueryHandler(self.captcha_button_callback, pattern=r"^captcha_button:"))
        self.application.add_handler(CallbackQueryHandler(self.captcha_math_callback, pattern=r"^captcha_math:"))
        self.application.add_handler(CallbackQueryHandler(self.refinement_callback, pattern=r"^refine:"))
        self.application.add_handler(CallbackQueryHandler(self.community_moderation.on_report_reason, pattern=r"^REPORT:"))
        
        conv_handler = ConversationHandler(
            entry_points=[
                MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.FORWARDED, self.handle_message),
                CommandHandler("news", self.news_command_entry)
            ],
            states={
                ASK_CITY: [MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.FORWARDED, self.handle_message)],
                ASK_NEWS_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.FORWARDED, self.handle_message)],
            },
            fallbacks=[CommandHandler("cancel", self.cancel_conversation)],
            allow_reentry=True
        )
        self.application.add_handler(conv_handler, group=50)
        # Register news command metadata for help and command list
        self.default_commands["news"] = {
            "short": "Get the latest news",
            "long": "Get the latest news",
        }

        # Add the standalone handlers for voice and video note messages
        self.application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self.handle_photo_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice_message))
        self.application.add_handler(MessageHandler(filters.VIDEO_NOTE, self.handle_video_note_message))

        # Register reaction handler only if message reactions are supported
        if MessageReactionHandler is not None:
            # Ensure reaction/refinement handlers are bound before registering
            self.bind_reaction_handlers()
            self.application.add_handler(MessageReactionHandler(self.reaction_router))
        else:
            logger.info(
                "Message reactions are not supported by this version of python-telegram-bot; "
                "reaction handler not registered."
            )

        self.application.add_error_handler(self.error_handler)
        logger.info("TelegramHandlerService: All handlers registered.")

        # Bind reaction handlers after all methods are set up
        self.bind_reaction_handlers()

    async def karmadiag_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return
        # Run diagnostics and echo a brief summary
        summary = await self.karma_manager.diagnose(chat_id)
        if update.message:
            await update.message.reply_text(summary)
        else:
            await context.bot.send_message(chat_id=chat_id, text=summary)
