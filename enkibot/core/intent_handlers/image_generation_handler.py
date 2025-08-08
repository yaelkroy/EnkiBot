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

# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
import logging 
import base64 
import re
from typing import Optional, TYPE_CHECKING

from telegram import Update
# InputFile might not be strictly necessary if only sending URLs or bytes for photos
from telegram.ext import ContextTypes 
from telegram.constants import ChatAction

from enkibot import config # Import config directly

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
        else:
            logger.error("Image generation prompt extractor prompts are missing or malformed (expecting 'user_template')!")
            # Basic fallback prompt cleaning
            bot_nicknames_to_check = config.BOT_NICKNAMES_TO_CHECK 
            bot_name_pattern = r"(?i)\b(?:{})\b\s*[:,]?\s*".format("|".join(re.escape(name) for name in bot_nicknames_to_check))
            cleaned_text_intermediate = re.sub(bot_name_pattern, "", user_msg_txt, count=1).strip()
            triggers_to_remove = ["draw", "create", "generate", "image of", "picture of", "сделай картинку", "нарисуй", "создай", "сгенерируй", "картинку", "изображение"]
            for trigger in triggers_to_remove:
                cleaned_text_intermediate = re.sub(r'(?i)\b' + re.escape(trigger) + r'\b\s*', '', cleaned_text_intermediate, count=1).strip()
            cleaned_text_intermediate = re.sub(r"^(can you|could you|please|i want|i need|дай мне|хочу)\s*", "", cleaned_text_intermediate, flags=re.IGNORECASE).strip()
            cleaned_text_intermediate = re.sub(r"[?.!]$", "", cleaned_text_intermediate).strip()
            if cleaned_text_intermediate and len(cleaned_text_intermediate) > 2: 
                 clean_prompt = cleaned_text_intermediate
            else:
                 logger.warning("Fallback prompt cleaning also resulted in no usable prompt.")


        if not clean_prompt:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=preliminary_reply.message_id,
                text=self.language_service.get_response_string("image_generation_no_prompt")
            )
            return

        try:
            generated_images_data = await self.llm_services.generate_image_openai(
                prompt=clean_prompt,
                n=config.DEFAULT_IMAGE_N,
                size=config.DEFAULT_IMAGE_SIZE,
                quality=config.DEFAULT_IMAGE_QUALITY,
                response_format="url"
            )
            
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preliminary_reply.message_id)

            if generated_images_data:
                caption_key = "image_generation_success_single" if len(generated_images_data) == 1 else "image_generation_success_multiple"
                
                for i, img_data in enumerate(generated_images_data):
                    current_caption = self.language_service.get_response_string(caption_key, image_prompt=clean_prompt) if i == 0 else None
                    if img_data.get("url"):
                        logger.info(f"DALL-E API returned a URL. Sending photo to user.")
                        await update.message.reply_photo(photo=img_data["url"], caption=current_caption)
                    elif img_data.get("b64_json"): 
                        logger.info("Image data returned as b64_json. Decoding and sending photo.")
                        image_bytes = base64.b64decode(img_data["b64_json"])
                        await update.message.reply_photo(photo=image_bytes, caption=current_caption)
                    else: 
                        logger.error(f"Image generation successful but no URL or b64_json data found for prompt: {clean_prompt}")
                        await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))

            else: 
                logger.error(f"Image generation failed for prompt: {clean_prompt} (no data returned from service or service call failed)")
                await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))

        except Exception as e:
            logger.error(f"An exception occurred while handling image generation: {e}", exc_info=True)
            try:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=preliminary_reply.message_id,
                    text=self.language_service.get_response_string("image_generation_error")
                )
            except Exception as edit_e: 
                logger.error(f"Failed to edit preliminary message during image error handling: {edit_e}")
                await update.message.reply_text(self.language_service.get_response_string("image_generation_error"))
