# enkibot/modules/image_generation_service.py
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

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

# This import makes the image generation tool available to be called.
from TBD_tool_name import image_generation

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.modules.intent_recognizer import IntentRecognizer

logger = logging.getLogger(__name__)

class ImageGenerationIntentHandler:
    def __init__(self, 
                 language_service: 'LanguageService', 
                 intent_recognizer: 'IntentRecognizer'):
        logger.info("ImageGenerationIntentHandler initialized.")
        self.language_service = language_service
        self.intent_recognizer = intent_recognizer

    async def handle_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> None:
        """
        Handles the full workflow for an image generation request, from prompt
        extraction to sending the final image or an error message.
        """
        if not update.message or not update.effective_chat:
            return

        # 1. Acknowledge the request and send a "typing" or "uploading" action.
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        # Send a preliminary "working on it" message that we can edit or delete later.
        preliminary_reply = await update.message.reply_text(self.language_service.get_response_string("image_generation_start"))

        # 2. Extract a clean, descriptive prompt from the user's full message.
        extractor_prompts = self.language_service.get_llm_prompt_set("image_generation_prompt_extractor")
        clean_prompt = None
        if extractor_prompts and "system" in extractor_prompts:
            clean_prompt = await self.intent_recognizer.extract_image_prompt_with_llm(
                text=user_msg_txt,
                lang_code=self.language_service.current_lang,
                system_prompt=extractor_prompts["system"],
                user_prompt_template=extractor_prompts.get("user", "{text}")
            )
        else:
            logger.error("Image generation prompt extractor prompts are missing!")

        # 3. Handle cases where a clear prompt could not be extracted.
        if not clean_prompt:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=preliminary_reply.message_id,
                text=self.language_service.get_response_string("image_generation_no_prompt")
            )
            return

        # 4. Call the image generation tool and handle the result.
        try:
            # This is the direct call to the image generation tool.
            image_gen_result = image_generation.generate_images(
                prompts=[clean_prompt],
                image_generation_usecase=image_generation.ImageGenerationUsecase.ALTERNATIVES
            )
            
            content_id = None
            if image_gen_result and image_gen_result.results:
                first_result = image_gen_result.results[0]
                if first_result and first_result.generated_images:
                    content_id = first_result.content_id

            # Delete the preliminary "On it! Imagining something..." message.
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preliminary_reply.message_id)

            if content_id:
                # If successful, send the image by replying with its content_id.
                # The Telegram client will render this as the image.
                logger.info(f"Successfully generated image for prompt '{clean_prompt}'. Replying with content_id.")
                await update.message.reply_text(content_id)
            else:
                # Handle the case where the tool ran but failed to produce an image.
                logger.error(f"Image generation tool ran but failed for prompt: {clean_prompt}")
                await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))

        except Exception as e:
            logger.error(f"An exception occurred during the image generation tool call: {e}", exc_info=True)
            # Edit the preliminary message to show an error instead of leaving the user hanging.
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=preliminary_reply.message_id,
                text=self.language_service.get_response_string("image_generation_error")
            )
