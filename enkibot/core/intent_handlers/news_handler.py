# enkibot/core/intent_handlers/news_handler.py
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
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler
from telegram.constants import ChatAction

if TYPE_CHECKING:
    from enkibot.core.language_service import LanguageService
    from enkibot.modules.intent_recognizer import IntentRecognizer
    from enkibot.modules.api_router import ApiRouter
    from enkibot.modules.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

# This state constant should ideally be defined in or imported from a central location
# to avoid magic numbers. For now, ensure it matches telegram_handlers.py.
ASK_NEWS_TOPIC = 2 

class NewsIntentHandler:
    def __init__(self, 
                 language_service: 'LanguageService', 
                 intent_recognizer: 'IntentRecognizer',
                 api_router: 'ApiRouter',
                 response_generator: 'ResponseGenerator',
                 pending_action_data_ref: Dict[int, Dict[str, Any]]):
        logger.info("NewsIntentHandler initialized.")
        self.language_service = language_service
        self.intent_recognizer = intent_recognizer
        self.api_router = api_router
        self.response_generator = response_generator
        self.pending_action_data = pending_action_data_ref # Reference to shared dict

    async def _process_news_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, topic: Optional[str], original_message_id: Optional[int] = None) -> int:
        """
        Fetches, compiles, and sends news. Central logic after topic is known.
        Returns ConversationHandler.END.
        """
        if not update.message or not update.effective_chat: 
            logger.warning("NewsHandler._process_news_request called with invalid update/context.")
            return ConversationHandler.END

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        logger.info(f"NewsHandler: Processing news for topic: '{topic if topic else 'General'}'")
        reply_to_id = original_message_id or update.message.message_id
        
        try:
            articles_structured = await self.api_router.get_latest_news_structured(
                query=topic, 
                lang_code=self.language_service.current_lang
            )

            if articles_structured is None: # API call itself failed
                logger.error(f"NewsHandler: API call to get_latest_news_structured failed for topic '{topic}'.")
                await update.message.reply_text(self.language_service.get_response_string("news_api_error"), reply_to_message_id=reply_to_id)
            elif not articles_structured: # API call succeeded but no articles found
                no_articles_key = "news_api_no_articles" if topic else "news_api_no_general_articles"
                await update.message.reply_text(self.language_service.get_response_string(no_articles_key, query=topic or ""), reply_to_message_id=reply_to_id)
            else: # Articles found, proceed to compile
                compiler_prompts = self.language_service.get_llm_prompt_set("news_compiler")
                if not (compiler_prompts and "system" in compiler_prompts and compiler_prompts.get("user_template")):
                    logger.error("NewsHandler: News compiler LLM prompts are missing or malformed (expecting 'user_template').")
                    await update.message.reply_text(self.language_service.get_response_string("news_api_data_error"), reply_to_message_id=reply_to_id)
                else:
                    compiled_response = await self.response_generator.compile_news_response(
                        articles_structured=articles_structured, topic=topic,
                        lang_code=self.language_service.current_lang,
                        system_prompt=compiler_prompts["system"],
                        user_prompt_template=compiler_prompts["user_template"]
                    )
                    await update.message.reply_text(compiled_response, disable_web_page_preview=True, reply_to_message_id=reply_to_id)
        
        except Exception as e:
            logger.error(f"NewsHandler: Unhandled exception in _process_news_request for topic '{topic}': {e}", exc_info=True)
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"), reply_to_message_id=reply_to_id)
        
        finally: 
            # Clean up conversation state regardless of success or failure in processing
            chat_id = update.effective_chat.id
            if chat_id in self.pending_action_data and \
               self.pending_action_data[chat_id].get("action_type") == "ask_news_topic":
                del self.pending_action_data[chat_id]
            if context.user_data and context.user_data.get('conversation_state') == ASK_NEWS_TOPIC:
                context.user_data.pop('conversation_state')
            return ConversationHandler.END

    async def handle_intent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_msg_txt: str) -> Optional[int]:
        """
        Handles an initial news intent (triggered by master_intent == NEWS_QUERY).
        Extracts topic or asks for it.
        """
        logger.info(f"NewsHandler: Handling initial NEWS_QUERY: '{user_msg_txt}'")
        if not update.message or not update.effective_chat: return ConversationHandler.END

        news_topic_prompts = self.language_service.get_llm_prompt_set("news_topic_extractor")
        if not (news_topic_prompts and "system" in news_topic_prompts and news_topic_prompts.get("user_template")): 
            logger.error("NewsHandler: News topic extractor LLM prompts missing or malformed (expecting 'user_template').")
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
            return ConversationHandler.END
        
        topic = await self.intent_recognizer.extract_news_topic_with_llm( 
            text=user_msg_txt, 
            lang_code=self.language_service.current_lang, 
            system_prompt=news_topic_prompts["system"], 
            user_prompt_template=news_topic_prompts["user_template"] 
        )

        if topic: 
            return await self._process_news_request(update, context, topic, update.message.message_id)
        else: 
            logger.info("NewsHandler: No topic identified from initial query, asking user.")
            self.pending_action_data[update.effective_chat.id] = {
                "action_type": "ask_news_topic",
                "original_message_id": update.message.message_id
            }
            context.user_data['conversation_state'] = ASK_NEWS_TOPIC # Ensure state is set in PTB
            await update.message.reply_text(self.language_service.get_response_string("news_ask_topic"))
            return ASK_NEWS_TOPIC

    async def handle_command_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        """
        Handles the /news command. This will always ask for a topic.
        """
        logger.info(f"NewsHandler: /news command received by user {update.effective_user.id if update.effective_user else 'Unknown'}")
        if not update.message or not update.effective_chat or not update.effective_user: 
            return ConversationHandler.END 
        
        # For /news, directly ask for topic.
        self.pending_action_data[update.effective_chat.id] = {
            "action_type": "ask_news_topic",
            "original_message_id": update.message.message_id
        }
        context.user_data['conversation_state'] = ASK_NEWS_TOPIC # Ensure state is set in PTB
        await update.message.reply_text(self.language_service.get_response_string("news_ask_topic"))
        return ASK_NEWS_TOPIC

    async def handle_topic_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE, original_message_id: Optional[int]) -> int:
        """
        Handles the user's response when they provide a news topic after being asked.
        """
        if not update.message or not update.message.text:
            # If user sends something non-text (e.g., sticker) or empty message
            await update.message.reply_text(self.language_service.get_response_string("news_ask_topic")) # Re-ask
            return ASK_NEWS_TOPIC # Remain in the same state

        user_reply_topic_text = update.message.text.strip()
        logger.info(f"NewsHandler: Received news topic reply '{user_reply_topic_text}' in ASK_NEWS_TOPIC state.")

        extractor_prompts = self.language_service.get_llm_prompt_set("news_topic_reply_extractor")
        logger.debug(f"NewsHandler: news_topic_reply_extractor prompts: {extractor_prompts}") 
        
        if not (extractor_prompts and "system" in extractor_prompts and extractor_prompts.get("user_template")):
            logger.error("NewsHandler: News topic reply extractor LLM prompts are missing or malformed (expecting 'user_template').")
            await update.message.reply_text(self.language_service.get_response_string("generic_error_message"))
            return ConversationHandler.END

        extracted_topic = await self.intent_recognizer.extract_topic_from_reply(
            text=user_reply_topic_text,
            lang_code=self.language_service.current_lang,
            system_prompt=extractor_prompts["system"],
            user_prompt_template=extractor_prompts["user_template"] 
        )

        # Use the extracted topic if successful, otherwise fall back to the user's raw input.
        # This handles cases where the LLM might return "None" or fail, but the user's text is still a valid topic.
        final_topic = extracted_topic or user_reply_topic_text
        logger.info(f"NewsHandler: Final topic for processing: '{final_topic}'")
        
        return await self._process_news_request(update, context, final_topic, original_message_id)
