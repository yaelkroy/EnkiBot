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

from telegram import Update
from telegram.ext import ContextTypes

from enkibot.modules.base_module import BaseModule
from enkibot.core.llm_services import LLMServices

logger = logging.getLogger(__name__)

class SpamDetector(BaseModule):
    """Detects spam or disallowed content and takes moderation actions."""

    def __init__(self, llm_services: LLMServices, enabled: bool = True):
        super().__init__("SpamDetector")
        self.llm_services = llm_services
        self.enabled = enabled
        logger.info("SpamDetector initialized. Enabled=%s", self.enabled)

    async def inspect_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Checks a message and removes it if flagged.

        Returns ``True`` if a message was flagged and handled (deleted/banned).
        """
        if not self.enabled:
            return False
        if not update.message or not update.message.text:
            return False

        moderation_result = await self.llm_services.moderate_text_openai(update.message.text)
        if not moderation_result or not moderation_result.get("flagged"):
            return False

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else None
        message_id = update.message.message_id

        # Delete offending message
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            logger.info("Deleted flagged message %s in chat %s", message_id, chat_id)
        except Exception as e:
            logger.error("Failed to delete flagged message %s: %s", message_id, e)

        # Ban user responsible for the message
        if user_id is not None:
            try:
                await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
                logger.info("Banned user %s for spam in chat %s", user_id, chat_id)
            except Exception as e:
                logger.error("Failed to ban user %s: %s", user_id, e)

        return True
