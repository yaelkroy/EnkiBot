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
# === EnkiBot Response Generator ===
# ==================================================================================================
# enkibot/modules/intent_recognizer.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)

# <<<--- DIAGNOSTIC PRINT IR-1: VERY TOP OF INTENT_RECOGNIZER.PY --- >>>
# print(f"%%%%% EXECUTING INTENT_RECOGNIZER.PY - VERSION FROM: {__file__} %%%%%") # You can uncomment this if you still face import issues

import logging
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from telegram.ext import ContextTypes 

# TYPE_CHECKING imports to avoid circular dependencies at runtime
if TYPE_CHECKING:
    from enkibot.core.llm_services import LLMServices
    from enkibot.utils.database import DatabaseManager
    from enkibot.modules.intent_recognizer import IntentRecognizer 
    # LanguageService is not directly used here, prompts are passed in

from enkibot.modules.fact_extractor import find_user_search_query_in_text 
from enkibot import config # For model IDs

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, 
                 llm_services: 'LLMServices', 
                 db_manager: 'DatabaseManager', 
                 intent_recognizer: 'IntentRecognizer'): # No LanguageService needed directly
        logger.info("ResponseGenerator __init__ STARTING")
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.intent_recognizer = intent_recognizer 
        logger.info("ResponseGenerator __init__ COMPLETED")

    async def analyze_replied_message(self, 
                                      original_text: str, 
                                      user_question: str,
                                      system_prompt: str, 
                                      user_prompt_template: Optional[str]) -> str:
        logger.info(f"ResponseGenerator: Analyzing replied message. Original length: {len(original_text)}, Question: '{user_question}'")
        
        user_prompt = (user_prompt_template or "Original Text:\n---\n{original_text}\n---\n\nUser's Question:\n---\n{user_question}\n---\n\nYour Analysis:").format(
            original_text=original_text, user_question=user_question
        )
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        try:
            # Use a general purpose model for this analysis task, or a specific one if configured
            analysis_model_id = config.OPENAI_MODEL_ID # Default general model
            
            if self.llm_services.is_provider_configured("openai"): 
                completion = await self.llm_services.call_openai_llm(
                    messages_for_api, 
                    model_id=analysis_model_id, 
                    temperature=0.5, 
                    max_tokens=1000
                )
                if completion: 
                    return completion.strip()
            
            # Fallback message if OpenAI is not configured or call fails
            logger.warning("OpenAI not configured or call failed for replied message analysis.")
            # This response should ideally come from language_service via the calling handler
            return "I'm having trouble analyzing that right now. Please try again later."
            
        except Exception as e:
            logger.error(f"Error in LLM call during replied message analysis: {e}", exc_info=True)
            # This response should ideally come from language_service via the calling handler
            return "Sorry, an error occurred during the text analysis."

    async def compile_weather_forecast_response(self, 
                                                forecast_data_structured: Dict[str, Any], 
                                                lang_code: str, # For logging and potentially Babel
                                                system_prompt: str, # Already localized and formatted by caller
                                                user_prompt_template: str) -> str:
        location = forecast_data_structured.get("location", "the requested location")
        forecast_data_json_str = json.dumps(forecast_data_structured.get("forecast_days", []), indent=2, ensure_ascii=False)

        user_prompt = user_prompt_template.format(
            location=location, 
            forecast_data_json=forecast_data_json_str
        )
        # System prompt is assumed to be already formatted with language_name by the caller (TelegramHandlerService)
            
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        logger.info(f"Requesting LLM to compile weather forecast for {location} in {lang_code}")
        # Use general model for compilation, or configure a specific one if desired
        compilation_model_id = config.OPENAI_MODEL_ID 
        compiled_response = await self.llm_services.race_llm_calls(messages) # Pass specific model if needed

        return compiled_response or "I found some weather data, but couldn't summarize it for you right now."


    async def compile_news_response(self, 
                                    articles_structured: List[Dict[str, Any]], 
                                    topic: Optional[str], 
                                    lang_code: str, # For logging and potentially Babel
                                    system_prompt: str, # Already localized and formatted by caller
                                    user_prompt_template: str) -> str:
        
        articles_json_str = json.dumps(articles_structured, indent=2, ensure_ascii=False)
        display_topic = topic if topic else "general interest"
            
        user_prompt = user_prompt_template.format(
            topic=display_topic, 
            articles_json=articles_json_str
        )
        # System prompt is assumed to be already formatted with language_name and topic by the caller
            
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        logger.info(f"Requesting LLM to compile news for topic '{display_topic}' in {lang_code}")
        # Use general model for compilation
        compilation_model_id = config.OPENAI_MODEL_ID
        compiled_response = await self.llm_services.race_llm_calls(messages) # Pass specific model if needed

        return compiled_response or "I found some news articles, but couldn't summarize them for you at the moment."

    async def get_orchestrated_llm_response(
        self, prompt_text: str, chat_id: int, user_id: int, message_id: int,
        context: ContextTypes.DEFAULT_TYPE, lang_code: str,
        system_prompt_override: str, 
        user_search_ambiguous_response_template: str,
        user_search_not_found_response_template: str
    ) -> str:
        logger.info(f"Orchestrating LLM response for: '{prompt_text[:100]}...' in chat {chat_id} (lang: {lang_code})")
        
        history_from_db: List[Dict[str, str]] = []
        profile_context_messages: List[Dict[str, str]] = []
        keyword_context_messages: List[Dict[str, str]] = [] 

        history_from_db = await self.db_manager.get_conversation_history(chat_id, limit=10) # Fetch recent history

        search_term_original = find_user_search_query_in_text(prompt_text) 
        
        if search_term_original:
            logger.info(f"User profile query detected for term: '{search_term_original}'")
            matched_profiles_data = await self.db_manager.find_user_profiles_by_name_variation(search_term_original)

            if len(matched_profiles_data) == 1:
                profile_data = matched_profiles_data[0]
                target_user_id_found = profile_data.get("UserID") 
                user_identifier = profile_data.get("FirstName") or profile_data.get("Username") or f"User ID {target_user_id_found}"

                if profile_data.get("Notes"): 
                    profile_context_messages.append({
                        "role": "system",
                        "content": f"Important Context (User Dossier) for '{user_identifier}':\n---\n{profile_data['Notes']}\n---"
                    })
                
                if target_user_id_found: 
                    user_specific_messages = await self.db_manager.get_user_messages_from_chat_log(target_user_id_found, chat_id, limit=5)
                    if user_specific_messages:
                        valid_user_messages = [msg for msg in user_specific_messages if isinstance(msg, str) and msg.strip()]
                        if valid_user_messages:
                            formatted_msgs = "\n".join([f'- "{msg_text}"' for msg_text in valid_user_messages])
                            keyword_context_messages.append({
                                "role": "system",
                                "content": f"Additional Context (recent raw messages) from '{user_identifier}'. Use with dossier for freshest insights:\n---\n{formatted_msgs}\n---"
                            })
            elif len(matched_profiles_data) > 1:
                user_options = [ f"@{p.get('Username')}" if p.get('Username') else f"{p.get('FirstName') or ''} {p.get('LastName') or ''}".strip() for p in matched_profiles_data ]
                user_options_str = ", ".join(filter(None, user_options))
                # The response string fetching should happen in TelegramHandlerService
                return user_search_ambiguous_response_template.format(user_options=user_options_str)
            else: 
                logger.info(f"No specific user profile found for search term '{search_term_original}'.")
                # Potentially return a "not found" message here if the query was ONLY about a user
                # For now, it will proceed to general LLM call which might still use the search_term_original
        
        # Construct messages for the LLM
        messages_for_api = []
        if system_prompt_override and isinstance(system_prompt_override, str):
            messages_for_api.append({"role": "system", "content": system_prompt_override})
        else:
            logger.error(f"system_prompt_override is invalid or None: {system_prompt_override}. Using a default.")
            messages_for_api.append({"role": "system", "content": "You are a helpful AI assistant."}) # Basic default

        messages_for_api.extend(profile_context_messages) 
        messages_for_api.extend(keyword_context_messages) 
        # Add general conversation history, ensuring not to duplicate if already part of keyword context
        # Simple approach: just add it. LLM should handle some redundancy.
        messages_for_api.extend(history_from_db) 
        
        if prompt_text and isinstance(prompt_text, str):
            messages_for_api.append({"role": "user", "content": prompt_text})
        else: # Should not happen if called from handle_message
            logger.error(f"prompt_text is invalid or None: {prompt_text}. Appending a placeholder.")
            messages_for_api.append({"role": "user", "content": "Please provide a general response."})

        # Context Truncation (simplified)
        MAX_TOTAL_MESSAGES_FOR_LLM = 20 # Example limit
        if len(messages_for_api) > MAX_TOTAL_MESSAGES_FOR_LLM:
            system_msgs = [m for m in messages_for_api if m["role"] == "system"]
            user_assistant_msgs = [m for m in messages_for_api if m["role"] != "system"]
            
            num_user_assistant_to_keep = MAX_TOTAL_MESSAGES_FOR_LLM - len(system_msgs)
            if num_user_assistant_to_keep < 1: # Ensure at least one user/assistant message if possible
                num_user_assistant_to_keep = 1 
            
            messages_for_api = system_msgs + user_assistant_msgs[-num_user_assistant_to_keep:]
            logger.info(f"Message context truncated to {len(messages_for_api)} messages.")

        logger.debug(f"Final messages for API ({len(messages_for_api)}): {json.dumps(messages_for_api, indent=2, ensure_ascii=False)[:1000]}...")

        final_reply = await self.llm_services.race_llm_calls(messages_for_api)

        if not final_reply:
            # This response should ideally come from language_service via the calling handler
            final_reply = "I'm currently unable to get a response from my core AI. Please try again shortly."
            logger.error("All LLM providers failed in race_llm_calls. Final_reply set to fallback.")

        # Save to conversation history (user prompt and bot reply)
        await self.db_manager.save_to_conversation_history(
            chat_id, user_id, message_id, 'user', prompt_text
        )
        if context.bot: 
            await self.db_manager.save_to_conversation_history(
                chat_id, context.bot.id, None, 'assistant', final_reply
            )
        return final_reply
