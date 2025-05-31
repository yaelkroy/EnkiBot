# enkibot/core/intent_handlers/image_generation_handler.py
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
# enkibot/core/intent_handlers/image_generation_handler.py
# (Your GPLv3 Header)

import logging
import base64 # For decoding b64_json
from typing import Optional, TYPE_CHECKING

from telegram import Update, InputFile
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from enkibot import config 

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.modules.intent_recognizer import IntentRecognizer
    from enkibot.core.llm_services import LLMServices

logger = logging.getLogger(__name__)

class ImageGenerationIntentHandler:
    def __init__(self, 
                 language_service: 'LanguageService', 
                 intent_recognizer: 'IntentRecognizer',
                 llm_services: 'LLMServices'):
        logger.info("ImageGenerationIntentHandler initialized.")
        self.language_service = language_service
        self.intent_recognizer = intent_recognizer
        self.llm_services = llm_services

    async def handle_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> None:
        if not update.message or not update.effective_chat:
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        preliminary_reply = await update.message.reply_text(
            self.language_service.get_response_string("image_generation_start")
        )

        extractor_prompts = self.language_service.get_llm_prompt_set("image_generation_prompt_extractor")
        clean_prompt: Optional[str] = None

        if extractor_prompts and "system" in extractor_prompts and extractor_prompts.get("user_template"):
            clean_prompt = await self.intent_recognizer.extract_image_prompt_with_llm(
                text=user_msg_txt,
                lang_code=self.language_service.current_lang,
                system_prompt=extractor_prompts["system"],
                user_prompt_template=extractor_prompts["user_template"]
            )
        # ... (fallback prompt cleaning if needed, as in previous version) ...

        if not clean_prompt:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=preliminary_reply.message_id,
                text=self.language_service.get_response_string("image_generation_no_prompt")
            )
            return

        try:
            # Using the new method for Responses API
            generated_images_data = await self.llm_services.generate_image_with_dalle(
                prompt=clean_prompt,
                n=config.DEFAULT_IMAGE_N,          # Use 'config' directly
                size=config.DEFAULT_IMAGE_SIZE,    # Use 'config' directly
                quality=config.DEFAULT_IMAGE_QUALITY, # Use 'config' directly
                response_format="url"
            )

            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preliminary_reply.message_id)

            if generated_images_data:
                # Assuming the Responses API tool call generates one image at a time per tool definition
                img_data = generated_images_data[0] # Get the first (and likely only) image data

                if img_data.get("b64_json"):
                    try:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                        # PTB's send_photo can take bytes directly for the photo argument
                        await update.message.reply_photo(
                            photo=image_bytes, 
                            caption=self.language_service.get_response_string("image_generation_success_single", image_prompt=clean_prompt)
                        )
                        logger.info(f"Successfully sent image generated from b64_json for prompt '{clean_prompt}'.")
                    except Exception as e:
                        logger.error(f"Failed to decode/send image from b64_json: {e}", exc_info=True)
                        await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))
                elif img_data.get("url"): # Should not happen with Responses API tool output example, but good to have a fallback
                    logger.warning("Image generation via Responses API returned a URL, expected b64_json. Attempting to send URL.")
                    await update.message.reply_photo(
                        photo=img_data["url"],
                        caption=self.language_service.get_response_string("image_generation_success_single", image_prompt=clean_prompt)
                    )
                else:
                    logger.error(f"Image generation failed (no b64_json or url) for prompt: {clean_prompt}")
                    await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))
            else:
                logger.error(f"Image generation failed for prompt: {clean_prompt} (no data returned from service)")
                await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))

        except Exception as e:
            logger.error(f"An exception occurred while handling image generation: {e}", exc_info=True)
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=preliminary_reply.message_id,
                    text=self.language_service.get_response_string("image_generation_error")
                )
            except Exception: 
                await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))