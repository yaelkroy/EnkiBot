# enkibot/modules/spam_detector.py
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

"""Spam detection module using OpenAI's moderation API."""

import logging
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes

from enkibot.modules.base_module import BaseModule
from enkibot.core.llm_services import LLMServices
from enkibot.utils.database import DatabaseManager

logger = logging.getLogger(__name__)


class SpamDetector(BaseModule):
    """Detects spam or disallowed content and takes moderation actions."""

    def __init__(
        self,
        llm_services: LLMServices,
        db_manager: Optional[DatabaseManager] = None,
        enabled: bool = True,
    ):
        super().__init__("SpamDetector")
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.enabled = enabled
        logger.info("SpamDetector initialized. Enabled=%s", self.enabled)

    async def inspect_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Checks a message and removes it if flagged.

        Returns ``True`` if a message was flagged and handled (deleted/banned).
        """
        if not self.enabled:
            return False
        if not update.message or not update.message.text:
            return False

        moderation_result = await self.llm_services.moderate_text_openai(
            update.message.text
        )
        if not moderation_result or not moderation_result.get("flagged"):
            return False

        chat = update.effective_chat
        chat_id = chat.id
        chat_name = (
            chat.title
            or getattr(chat, "full_name", None)
            or chat.username
            or str(chat_id)
        )
        user = update.effective_user
        user_id = user.id if user else None
        user_name = (user.full_name if user else None) or getattr(
            user, "username", "Unknown"
        )
        message_id = update.message.message_id
        bot_name = context.bot.username or str(context.bot.id)

        # Log moderation action to database if configured
        if self.db_manager and self.db_manager.connection_string:
            try:
                categories = moderation_result.get("categories", {})
                flagged = ", ".join([k for k, v in categories.items() if v])
                await self.db_manager.log_moderation_action(
                    chat_id=chat_id,
                    user_id=user_id,
                    message_id=message_id,
                    categories=flagged or None,
                )
            except Exception as e:  # pragma: no cover - logging shouldn't raise
                logger.error("Failed to log moderation action: %s", e, exc_info=True)

        # Delete offending message
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            logger.info(
                "Bot %s deleted flagged message %s from %s (%s) in chat %s (%s)",
                bot_name,
                message_id,
                user_name,
                user_id,
                chat_name,
                chat_id,
            )
        except Exception as e:
            logger.error(
                "Bot %s failed to delete flagged message %s from %s (%s) in chat %s (%s): %s",
                bot_name,
                message_id,
                user_name,
                user_id,
                chat_name,
                chat_id,
                e,
            )

        # Ban user responsible for the message
        if user_id is not None:
            try:
                member = await context.bot.get_chat_member(
                    chat_id=chat_id, user_id=user_id
                )
            except Exception as e:
                logger.error(
                    "Bot %s failed to fetch member info for user %s (%s) in chat %s (%s): %s",
                    bot_name,
                    user_name,
                    user_id,
                    chat_name,
                    chat_id,
                    e,
                )
            else:
                user_name = (
                    member.user.full_name
                    or getattr(member.user, "username", None)
                    or str(user_id)
                )
                if member.status == "creator":
                    logger.warning(
                        "Bot %s cannot ban chat owner %s (%s) in chat %s (%s)",
                        bot_name,
                        user_name,
                        user_id,
                        chat_name,
                        chat_id,
                    )
                elif member.status == "administrator":
                    try:
                        bot_member = await context.bot.get_chat_member(
                            chat_id=chat_id, user_id=context.bot.id
                        )
                        if bot_member.status == "creator":
                            await context.bot.ban_chat_member(
                                chat_id=chat_id, user_id=user_id
                            )
                            logger.info(
                                "Bot %s banned admin %s (%s) for spam in chat %s (%s)",
                                bot_name,
                                user_name,
                                user_id,
                                chat_name,
                                chat_id,
                            )
                        else:
                            logger.warning(
                                "Bot %s lacks rights to ban admin %s (%s) in chat %s (%s)",
                                bot_name,
                                user_name,
                                user_id,
                                chat_name,
                                chat_id,
                            )
                    except Exception as e:
                        logger.error(
                            "Bot %s failed to ban admin %s (%s) in chat %s (%s): %s",
                            bot_name,
                            user_name,
                            user_id,
                            chat_name,
                            chat_id,
                            e,
                        )
                else:
                    try:
                        await context.bot.ban_chat_member(
                            chat_id=chat_id, user_id=user_id
                        )
                        logger.info(
                            "Bot %s banned user %s (%s) for spam in chat %s (%s)",
                            bot_name,
                            user_name,
                            user_id,
                            chat_name,
                            chat_id,
                        )
                    except Exception as e:
                        logger.error(
                            "Bot %s failed to ban user %s (%s) in chat %s (%s): %s",
                            bot_name,
                            user_name,
                            user_id,
                            chat_name,
                            chat_id,
                            e,
                        )

        return True
