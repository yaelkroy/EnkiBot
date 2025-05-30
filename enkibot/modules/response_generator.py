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
import logging
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from telegram.ext import ContextTypes 

# Use TYPE_CHECKING for imports that could cause circular dependencies if imported directly
if TYPE_CHECKING:
    from enkibot.core.llm_services import LLMServices
    from enkibot.utils.database import DatabaseManager
    from enkibot.modules.intent_recognizer import IntentRecognizer 

from enkibot.modules.fact_extractor import find_user_search_query_in_text 

logger = logging.getLogger(__name__)

class ResponseGenerator: # Make sure this class definition is present and correct
    def __init__(self, llm_services: 'LLMServices', db_manager: 'DatabaseManager', intent_recognizer: 'IntentRecognizer'):
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.intent_recognizer = intent_recognizer 

    async def analyze_replied_message(self, original_text: str, user_question: str,
                                      system_prompt: str, user_prompt_template: Optional[str]) -> str:
        logger.info(f"Analyzing replied message. Original length: {len(original_text)}, Question: '{user_question}'")
        user_prompt = (user_prompt_template or "Original: {original_text}\nQuestion: {user_question}\nAnalysis:").format(
            original_text=original_text, user_question=user_question
        )
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            if self.llm_services.is_provider_configured("openai"): 
                completion = await self.llm_services.call_openai_llm(messages_for_api, temperature=0.5, max_tokens=1000)
                if completion: return completion.strip()
            # Fallback if OpenAI is not configured or call fails
            return "Analysis client not configured or error in API call." 
        except Exception as e:
            logger.error(f"Error in LLM call during replied message analysis: {e}", exc_info=True)
            return "Sorry, an error occurred during text analysis."

    async def get_orchestrated_llm_response(
        self, prompt_text: str, chat_id: int, user_id: int, message_id: int,
        context: ContextTypes.DEFAULT_TYPE, lang_code: str,
        system_prompt_override: str, 
        user_search_ambiguous_response_template: str,
        user_search_not_found_response_template: str
    ) -> str:
        logger.info(f"Orchestrating LLM response for: '{prompt_text[:100]}...' in chat {chat_id}")
        
        history_from_db: List[Dict[str, str]] = []
        profile_context_messages: List[Dict[str, str]] = []
        keyword_context_messages: List[Dict[str, str]] = [] 

        history_from_db = await self.db_manager.get_conversation_history(chat_id, limit=10)

        # Call the imported function directly
        search_term_original = find_user_search_query_in_text(prompt_text) 
        
        if search_term_original:
            logger.info(f"User profile query detected for term: '{search_term_original}' in get_orchestrated_llm_response")
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
                return user_search_ambiguous_response_template.format(user_options=user_options_str)
            else: 
                 logger.info(f"No specific user profile found for search term '{search_term_original}'.")
        
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
            logger.error(f"prompt_text is invalid or None: {prompt_text}. Appending a placeholder user message.")
            messages_for_api.append({"role": "user", "content": "Hello."})

        logger.info(f"Constructed messages_for_api for race_llm_calls. Total messages: {len(messages_for_api)}")
        for i, msg in enumerate(messages_for_api):
            role = msg.get("role")
            content_sample = str(msg.get("content"))[:70] + "..." if msg.get("content") else "None"
            logger.debug(f"  Msg {i}: Role='{role}', ContentSample='{content_sample}'")
            if not isinstance(msg.get("role"), str) or not isinstance(msg.get("content"), str):
                logger.error(f"INVALID MESSAGE PART DETECTED in messages_for_api at index {i}: Role type {type(msg.get('role'))}, Content type {type(msg.get('content'))}")
        
        MAX_CONTEXT_MSGS = 20 
        if len(messages_for_api) > MAX_CONTEXT_MSGS:
            system_prompts_initial = [m for m in messages_for_api if m.get("role") == "system"]
            other_messages = [m for m in messages_for_api if m.get("role") != "system"]
            num_other_to_keep = MAX_CONTEXT_MSGS - len(system_prompts_initial)
            if num_other_to_keep < 0: num_other_to_keep = 5 
            
            final_messages_for_api = system_prompts_initial + other_messages[-num_other_to_keep:]
            logger.info(f"Message context truncated from {len(messages_for_api)} to {len(final_messages_for_api)} messages.")
            messages_for_api = final_messages_for_api

        final_reply = await self.llm_services.race_llm_calls(messages_for_api)

        if not final_reply:
            final_reply = "I'm currently unable to get a response from my core AI. Please try again shortly."
            logger.error("All LLM providers failed in race_llm_calls. Final_reply set to fallback.")

        await self.db_manager.save_to_conversation_history(
            chat_id, user_id, message_id, 'user', prompt_text
        )
        if context.bot: 
            await self.db_manager.save_to_conversation_history(
                chat_id, context.bot.id, None, 'assistant', final_reply
            )
        return final_reply