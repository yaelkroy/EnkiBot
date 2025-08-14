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
from enkibot.utils.message_utils import is_forwarded_message
from enkibot.utils.trigger_extractor import extract_assistant_prompt

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
        self.reaction_handlers: Dict[str, Any] = {
            "🔄": self._handle_regenerate_reaction,
            "➕": self._handle_expand_reaction,
            "📝": self._handle_summary_reaction,
        }

        # Track recent reactions to bot messages for follow-up triggers
        self.recent_reactors: Dict[int, datetime] = {}

        # Map inline refinement actions to handlers
        self.refinement_handlers: Dict[str, Any] = {
            "regenerate": self._handle_regenerate_reaction,
            "expand": self._handle_expand_reaction,
            "summary": self._handle_summary_reaction,
        }

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
        """Handle inline refinement buttons under bot replies (Regenerate/Expand/Summarize)."""
        query = update.callback_query
        if not query or not query.data:
            return
        await query.answer()
        parts = query.data.split(":")
        if len(parts) != 2:
            return
        _prefix, action = parts
        try:
            await self._perform_refinement_from_callback(query, context, action)
        except Exception as exc:
            logger.error(f"Refinement failed: {exc}", exc_info=True)
            try:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,  # type: ignore[attr-defined]
                    text=self.language_service.get_response_string("generic_error_message", "Something went wrong."),
                )
            except Exception:
                pass

    async def _perform_refinement_from_callback(self, query, context: ContextTypes.DEFAULT_TYPE, action: str) -> None:
        """Generate refined output using the original user message and previous bot answer.

        - regenerate: produce an alternative answer to the same user message
        - expand: elaborate on the previous assistant answer
        - summary: summarize the previous assistant answer concisely
        """
        bot_msg = getattr(query, "message", None)
        if not bot_msg:
            return
        orig_user_msg = getattr(bot_msg, "reply_to_message", None)
        if not orig_user_msg:
            # If not a reply, fallback to expanding the bot message itself
            orig_user_text = None
        else:
            orig_user_text = (orig_user_msg.text or orig_user_msg.caption or "").strip()
        prior_answer_text = (bot_msg.text or bot_msg.caption or "").strip()
        if not prior_answer_text and not orig_user_text:
            return

        # Choose prompts per action
        if action == "regenerate":
            system_prompt = (
                "You are a helpful assistant. Provide an alternative answer to the user's message. "
                "Avoid repeating the phrasing of the previous assistant answer. Be clear and useful."
            )
            user_payload = (
                (f"User message:\n{orig_user_text}\n\nPrevious assistant answer:\n{prior_answer_text}\n\nAlternative answer:" if orig_user_text else prior_answer_text)
            )
        elif action == "expand":
            system_prompt = (
                "You are a helpful assistant. Expand and enrich the previous assistant answer to the user's message. "
                "Add helpful detail, structure, examples, and clarifications. Keep language consistent."
            )
            user_payload = (
                f"User message:\n{orig_user_text or ''}\n\nPrevious assistant answer (to expand):\n{prior_answer_text}\n\nExpanded answer:"
            )
        else:  # summary
            system_prompt = (
                "You are a helpful assistant. Summarize the assistant's previous answer into a concise, readable summary. "
                "Preserve key points. Keep it short."
            )
            user_payload = (
                f"Previous assistant answer:\n{prior_answer_text}\n\nConcise summary:"
            )

        # Indicate typing
        try:
            await context.bot.send_chat_action(chat_id=bot_msg.chat_id, action=ChatAction.TYPING)
        except Exception:
            pass

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        # Run via orchestrator provider race
        try:
            refined = await self.llm_services.race_llm_calls(messages)
        except Exception as e:
            logger.error(f"LLM refine call failed: {e}", exc_info=True)
            refined = None

        text = clean_output_text(refined or "")
        if not text:
            # Fallback minimal acknowledgment
            await context.bot.send_message(
                chat_id=bot_msg.chat_id,
                text=self.language_service.get_response_string("generic_error_message", "Couldn't refine this message."),
                reply_to_message_id=getattr(orig_user_msg, "message_id", None) or bot_msg.message_id,
            )
            return

        # Recreate the inline keyboard for further refinement
        keyboard = InlineKeyboardMarkup(
            [[
                InlineKeyboardButton("\U0001F504 Regenerate", callback_data="refine:regenerate"),
                InlineKeyboardButton("\u2795 Expand", callback_data="refine:expand"),
                InlineKeyboardButton("\U0001F4DD Summarize", callback_data="refine:summary"),
            ]
            ]
        )
        # Send possibly chunked output as a reply to the original user message if available
        chunks = split_text_into_chunks(text)
        target_reply_id = getattr(orig_user_msg, "message_id", None) or bot_msg.message_id
        for i, chunk in enumerate(chunks):
            await context.bot.send_message(
                chat_id=bot_msg.chat_id,
                text=chunk,
                reply_to_message_id=target_reply_id,
                reply_markup=keyboard if i == 0 else None,
            )
