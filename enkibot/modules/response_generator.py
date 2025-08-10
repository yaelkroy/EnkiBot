# enkibot/modules/response_generator.py
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
# ==================================================================================================
# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# === EnkiBot Response Generator ===
# ==================================================================================================
# enkibot/modules/intent_recognizer.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)

# <<<--- DIAGNOSTIC PRINT IR-1: VERY TOP OF INTENT_RECOGNIZER.PY --- >>>
# print(f"%%%%% EXECUTING INTENT_RECOGNIZER.PY - VERSION FROM: {__file__} %%%%%") # You can uncomment this if you still face import issues
# enkibot/modules/response_generator.py
# (Your GPLv3 Header)

import logging
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from telegram.ext import ContextTypes 

if TYPE_CHECKING:
    from enkibot.core.llm_services import LLMServices
    from enkibot.utils.database import DatabaseManager
    from enkibot.modules.intent_recognizer import IntentRecognizer 

from enkibot.modules.fact_extractor import find_user_search_query_in_text 
from enkibot import config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, 
                 llm_services: 'LLMServices', 
                 db_manager: 'DatabaseManager', 
                 intent_recognizer: 'IntentRecognizer'):
        logger.info("ResponseGenerator __init__ STARTING")
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.intent_recognizer = intent_recognizer 
        logger.info("ResponseGenerator __init__ COMPLETED")

    # ... (analyze_replied_message, compile_weather_forecast_response, compile_news_response, get_orchestrated_llm_response methods remain the same) ...
    async def analyze_replied_message(self, original_text: str, user_question: str, system_prompt: str, user_prompt_template: Optional[str]) -> str:
        logger.info(f"ResponseGenerator: Analyzing replied message. Length: {len(original_text)}, Q: '{user_question}'")
        user_prompt = (user_prompt_template or "Original Text:\n---\n{original_text}\n---\n\nUser's Question:\n---\n{user_question}\n---\n\nYour Analysis:").format(
            original_text=original_text, user_question=user_question )
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            analysis_model_id = config.OPENAI_MODEL_ID 
            if self.llm_services.is_provider_configured("openai"): 
                completion = await self.llm_services.call_openai_llm(messages_for_api, model_id=analysis_model_id, temperature=0.5, max_tokens=1000)
                if completion: return completion.strip()
            return "Analysis client not configured or error in API call."
        except Exception as e:
            logger.error(f"Error in LLM call during replied message analysis: {e}", exc_info=True)
            return "Sorry, an error occurred during the text analysis."

    async def fact_check_forwarded_message(self, forwarded_text: str, user_question: str,
                                           system_prompt: str, user_prompt_template: Optional[str]) -> str:
        """Performs a fact-check on a forwarded message using LLM deep-research capabilities."""
        logger.info(
            "ResponseGenerator: Fact-checking forwarded message. Length: %d, Q: '%s'",
            len(forwarded_text), user_question
        )
        user_prompt = (
            user_prompt_template or
            "Forwarded Text:\n---\n{forwarded_text}\n---\n\nUser's Question:\n---\n{user_question}\n---\n\nYour Fact-Check:"
        ).format(forwarded_text=forwarded_text, user_question=user_question)
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            if self.llm_services.is_provider_configured("openai"):
                fact_check_response = await self.llm_services.call_openai_llm(
                    messages_for_api,
                    model_id=self.llm_services.openai_deep_research_model_id,
                    temperature=0.0,
                    max_tokens=1000,
                )
            else:
                fact_check_response = await self.llm_services.race_llm_calls(messages_for_api)
            return fact_check_response or "I couldn't verify that message right now."
        except Exception as e:
            logger.error("Error in LLM call during forwarded message fact-check: %s", e, exc_info=True)
            return "Sorry, I couldn't verify that message."

    async def compile_weather_forecast_response(self, forecast_data_structured: Dict[str, Any], lang_code: str, system_prompt: str, user_prompt_template: str) -> str:
        location = forecast_data_structured.get("location", "the requested location")
        forecast_data_json_str = json.dumps(forecast_data_structured.get("forecast_days", []), indent=2, ensure_ascii=False)
        user_prompt = user_prompt_template.format(location=location, forecast_data_json=forecast_data_json_str)
        language_name_for_prompt = lang_code 
        try:
            from babel import Locale
            locale = Locale.parse(lang_code)
            language_name_for_prompt = locale.get_display_name('en') 
        except Exception: pass
        if "{location}" in system_prompt or "{language_name}" in system_prompt:
            system_prompt = system_prompt.format(location=location, language_name=language_name_for_prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.info(f"Requesting LLM to compile weather forecast for {location} in {lang_code}")
        compiled_response = await self.llm_services.race_llm_calls(messages)
        return compiled_response or "I found some weather data, but couldn't summarize it for you right now."

    async def compile_news_response(self, articles_structured: List[Dict[str, Any]], topic: Optional[str], lang_code: str, system_prompt: str, user_prompt_template: str) -> str:
        articles_json_str = json.dumps(articles_structured, indent=2, ensure_ascii=False)
        display_topic = topic if topic else "general interest"
        user_prompt = user_prompt_template.format(topic=display_topic, articles_json=articles_json_str)
        language_name_for_prompt = lang_code
        try:
            from babel import Locale
            locale = Locale.parse(lang_code)
            language_name_for_prompt = locale.get_display_name('en')
        except Exception: pass
        if "{topic}" in system_prompt or "{language_name}" in system_prompt:
            system_prompt = system_prompt.format(topic=display_topic, language_name=language_name_for_prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.info(f"Requesting LLM to compile news for topic '{display_topic}' in {lang_code}")
        compiled_response = await self.llm_services.race_llm_calls(messages)
        return compiled_response or "I found some news articles, but couldn't summarize them for you at the moment."

    async def get_orchestrated_llm_response(self, prompt_text: str, chat_id: int, user_id: int, message_id: int, context: ContextTypes.DEFAULT_TYPE, lang_code: str, system_prompt_override: str, user_search_ambiguous_response_template: str, user_search_not_found_response_template: str) -> str:
        logger.info(f"Orchestrating LLM response for: '{prompt_text[:100]}...' in chat {chat_id} (lang: {lang_code})")
        history_from_db: List[Dict[str, str]] = []
        profile_context_messages: List[Dict[str, str]] = []
        keyword_context_messages: List[Dict[str, str]] = [] 
        history_from_db = await self.db_manager.get_conversation_history(chat_id, limit=10)
        search_term_original = find_user_search_query_in_text(prompt_text) 
        if search_term_original:
            logger.info(f"User profile query detected for term: '{search_term_original}'")
            matched_profiles_data = await self.db_manager.find_user_profiles_by_name_variation(search_term_original)
            if len(matched_profiles_data) == 1:
                profile_data = matched_profiles_data[0]
                target_user_id_found = profile_data.get("UserID") 
                user_identifier = profile_data.get("FirstName") or profile_data.get("Username") or f"User ID {target_user_id_found}"
                if profile_data.get("Notes"): 
                    profile_context_messages.append({"role": "system", "content": f"Important Context (User Dossier) for '{user_identifier}':\n---\n{profile_data['Notes']}\n---"})
                if target_user_id_found: 
                    user_specific_messages = await self.db_manager.get_user_messages_from_chat_log(target_user_id_found, chat_id, limit=5)
                    if user_specific_messages:
                        valid_user_messages = [msg for msg in user_specific_messages if isinstance(msg, str) and msg.strip()]
                        if valid_user_messages:
                            formatted_msgs = "\n".join([f'- "{msg_text}"' for msg_text in valid_user_messages])
                            keyword_context_messages.append({"role": "system", "content": f"Additional Context (recent raw messages) from '{user_identifier}'. Use with dossier for freshest insights:\n---\n{formatted_msgs}\n---"})
            elif len(matched_profiles_data) > 1:
                user_options = [ f"@{p.get('Username')}" if p.get('Username') else f"{p.get('FirstName') or ''} {p.get('LastName') or ''}".strip() for p in matched_profiles_data ]
                user_options_str = ", ".join(filter(None, user_options))
                return user_search_ambiguous_response_template.format(user_options=user_options_str)
        messages_for_api = []
        if system_prompt_override and isinstance(system_prompt_override, str):
            messages_for_api.append({"role": "system", "content": system_prompt_override})
        else:
            logger.error(f"system_prompt_override is invalid or None: {system_prompt_override}. Using a default.")
            messages_for_api.append({"role": "system", "content": "You are a helpful AI assistant."})
        messages_for_api.extend(profile_context_messages) 
        messages_for_api.extend(keyword_context_messages) 
        messages_for_api.extend(history_from_db) 
        if prompt_text and isinstance(prompt_text, str):
            messages_for_api.append({"role": "user", "content": prompt_text})
        else:
            logger.error(f"prompt_text is invalid or None: {prompt_text}. Appending a placeholder.")
            messages_for_api.append({"role": "user", "content": "Please provide a general response."})
        MAX_CONTEXT_MSGS = 20 
        if len(messages_for_api) > MAX_CONTEXT_MSGS:
            system_prompts_initial = [m for m in messages_for_api if m.get("role") == "system"]
            other_messages = [m for m in messages_for_api if m.get("role") != "system"]
            num_other_to_keep = MAX_CONTEXT_MSGS - len(system_prompts_initial)
            if num_other_to_keep < 1: num_other_to_keep = 1 
            final_messages_for_api = system_prompts_initial + other_messages[-num_other_to_keep:]
            logger.info(f"Message context truncated from {len(messages_for_api)} to {len(final_messages_for_api)} messages.")
            messages_for_api = final_messages_for_api
        final_reply = await self.llm_services.race_llm_calls(messages_for_api)
        if not final_reply:
            final_reply = "I'm currently unable to get a response from my core AI. Please try again shortly."
            logger.error("All LLM providers failed in race_llm_calls. Final_reply set to fallback.")
        await self.db_manager.save_to_conversation_history(chat_id, user_id, message_id, 'user', prompt_text)
        if context.bot: 
            await self.db_manager.save_to_conversation_history(chat_id, context.bot.id, None, 'assistant', final_reply)
        return final_reply

    # --- NEW METHOD FOR TRANSLATION ---
    async def translate_text(self, 
                             text_to_translate: str, 
                             target_language: str,
                             system_prompt: str,
                             user_prompt_template: str) -> Optional[str]:
        """
        Uses an LLM to translate text to a specified target language.
        """
        logger.info(f"Requesting translation of text to '{target_language}': '{text_to_translate[:70]}...'")
        
        # Format the prompts with the target language and text
        system_prompt = system_prompt.format(target_language=target_language)
        user_prompt = user_prompt_template.format(text_to_translate=text_to_translate)
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        # Use a specific translation model or race general ones
        translation_model_id = self.llm_services.openai_translation_model_id
        
        try:
            # For translation, a single high-quality model is often better than racing
            if self.llm_services.is_provider_configured("openai"):
                translated_text = await self.llm_services.call_openai_llm(
                    messages,
                    model_id=translation_model_id,
                    temperature=0.1 # Low temperature for more literal translation
                )
                if translated_text:
                    logger.info("Translation successful.")
                    return translated_text
            else: # Fallback to racing if OpenAI is not available
                logger.warning("OpenAI not configured for translation, falling back to racing general models.")
                translated_text = await self.llm_services.race_llm_calls(messages)
                if translated_text:
                    logger.info("Translation successful via race.")
                    return translated_text

        except Exception as e:
            logger.error(f"An error occurred during translation: {e}", exc_info=True)
        
        logger.error("Translation failed for all providers.")
        return None
