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
# <<<--- DIAGNOSTIC PRINT IR-1: VERY TOP OF INTENT_RECOGNIZER.PY --- >>>
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)

# <<<--- DIAGNOSTIC PRINT IR-1: VERY TOP OF INTENT_RECOGNIZER.PY --- >>>
# enkibot/modules/intent_recognizer.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)
# enkibot/modules/intent_recognizer.py
# EnkiBot: Advanced Multilingual Telegram AI Assistant
# Copyright (C) 2025 Yael Demedetskaya <yaelkroy@gmail.com>
# (Your GPLv3 Header)

import logging
import json
import re # <--- ***** ADD THIS IMPORT *****
from typing import Dict, Any, Optional, TYPE_CHECKING

from enkibot import config # For model IDs

if TYPE_CHECKING:
    from enkibot.core.llm_services import LLMServices 

logger = logging.getLogger(__name__)

class IntentRecognizer:
    def __init__(self, llm_services: 'LLMServices'): 
        logger.info("IntentRecognizer __init__ STARTING")
        self.llm_services = llm_services
        logger.info("IntentRecognizer __init__ COMPLETED")

    async def classify_master_intent(self, text: str, lang_code: str, 
                                     system_prompt: str, user_prompt_template: str) -> str:
        logger.info(f"Classifying master intent (lang: {lang_code}): '{text[:100]}...'")
        user_prompt = user_prompt_template.format(text_to_classify=text)
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        response_format_arg = {}
        classification_model_id = config.OPENAI_CLASSIFICATION_MODEL_ID
        # Check if the classification_model_id is one known to support JSON object mode
        if classification_model_id and \
           any(model_prefix in classification_model_id for model_prefix in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]):
             response_format_arg = {"response_format": {"type": "json_object"}}

        classified_intent_value = "UNKNOWN_INTENT" 
        completion_str_for_log = "N/A"

        try:
            completion_str = await self.llm_services.call_openai_llm(
                messages_for_api, 
                model_id=classification_model_id, 
                temperature=0.0, 
                max_tokens=100,
                **response_format_arg 
            )
            completion_str_for_log = completion_str if completion_str is not None else "None"

            if completion_str:
                try:
                    clean_comp_str = completion_str.strip()
                    # More robustly remove markdown json block using re
                    match = re.search(r"```json\s*(.*?)\s*```", clean_comp_str, re.DOTALL | re.IGNORECASE)
                    if match:
                        clean_comp_str = match.group(1).strip()
                    elif clean_comp_str.startswith("```"): # Handle simple ``` case
                        clean_comp_str = clean_comp_str.strip("` \t\n\r")
                        if clean_comp_str.lower().startswith("json"): # check if 'json' follows ```
                            clean_comp_str = clean_comp_str[4:].strip() # remove 'json' and strip
                    
                    data = json.loads(clean_comp_str)
                    intent_from_json = data.get("intent", data.get("INTENT")) 

                    if intent_from_json and isinstance(intent_from_json, str):
                        processed_intent = intent_from_json.strip().strip('_').upper().replace(" ", "_")
                        known_categories = ["WEATHER_QUERY", "NEWS_QUERY", "USER_PROFILE_QUERY", "MESSAGE_ANALYSIS_QUERY", "GENERAL_CHAT", "UNKNOWN_INTENT"]
                        if processed_intent in known_categories:
                            classified_intent_value = processed_intent
                            logger.info(f"Master intent classified as: {classified_intent_value} via JSON.")
                        else:
                            logger.warning(f"LLM JSON with unknown category '{processed_intent}'. Raw value: '{intent_from_json}'. Full data: {data}. Defaulting UNKNOWN.")
                    else:
                        logger.warning(f"LLM JSON for master intent missing 'intent' key or not string. Data: {data}. Defaulting UNKNOWN.")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON from master_intent_classifier. LLM raw: '{completion_str_for_log}'. Attempting direct parse.")
                    raw_intent = completion_str_for_log.strip().strip('_').upper().replace(" ", "_")
                    if raw_intent.startswith('{"INTENT":') and raw_intent.endswith('"}'):
                        try:
                            raw_intent_data = json.loads(raw_intent) # Try to parse this specific format
                            raw_intent = raw_intent_data.get("INTENT", raw_intent).strip().strip('_').upper().replace(" ", "_")
                        except: pass 
                    
                    known_categories = ["WEATHER_QUERY", "NEWS_QUERY", "USER_PROFILE_QUERY", "MESSAGE_ANALYSIS_QUERY", "GENERAL_CHAT", "UNKNOWN_INTENT"] 
                    if raw_intent in known_categories:
                        classified_intent_value = raw_intent
                        logger.info(f"Master intent classified as: {classified_intent_value} via direct string parse fallback.")
                    else:
                         logger.warning(f"Direct string parse fallback failed for intent: '{raw_intent}'. Defaulting UNKNOWN.")
            else:
                logger.warning("Master intent classification LLM call returned no content.")
        except Exception as e: 
            logger.error(f"Error during master intent classification LLM call: {e}", exc_info=True)
        
        return classified_intent_value

    async def analyze_weather_request_with_llm(self, text: str, lang_code: str, 
                                               system_prompt: str, user_prompt_template: Optional[str]) -> Dict[str, Any]:
        logger.info(f"Analyzing weather request type (lang: {lang_code}): '{text}' with LLM.")
        user_prompt = (user_prompt_template or "{text}").format(text=text) 
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        response_format_arg = {}
        model_to_use = config.OPENAI_CLASSIFICATION_MODEL_ID
        if model_to_use and any(model_prefix in model_to_use for model_prefix in ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]):
             response_format_arg = {"response_format": {"type": "json_object"}}
        
        default_response = {"type": "current"} 
        completion_str_for_error_log = "N/A"
        try:
            completion_str = await self.llm_services.call_openai_llm(
                messages_for_api, 
                model_id=model_to_use,
                temperature=0, 
                **response_format_arg
            )
            completion_str_for_error_log = completion_str if completion_str is not None else "None"
            if completion_str:
                logger.info(f"LLM response for weather analysis: {completion_str}")
                clean_comp_str = completion_str.strip()
                match = re.search(r"```json\s*(.*?)\s*```", clean_comp_str, re.DOTALL | re.IGNORECASE)
                if match: clean_comp_str = match.group(1).strip()
                else:
                    if clean_comp_str.startswith("```json"): clean_comp_str = clean_comp_str[7:]
                    if clean_comp_str.endswith("```"): clean_comp_str = clean_comp_str[:-3]                
                return json.loads(clean_comp_str.strip())
            else:
                logger.warning("LLM returned no content for weather analysis.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM for weather analysis: '{completion_str_for_error_log}'. Error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in LLM call during weather request analysis: {e}", exc_info=True)
        
        return default_response

    async def extract_location_with_llm(self, text: str, lang_code: str, 
                                        system_prompt: str, user_prompt_template: Optional[str]) -> Optional[str]:
        logger.info(f"Requesting LLM location extraction from text (lang: {lang_code}): '{text}'")
        user_prompt = (user_prompt_template or "{text}").format(text=text)
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        location = None
        model_to_use = config.OPENAI_CLASSIFICATION_MODEL_ID
        try:
            if self.llm_services.is_provider_configured("openai"):
                completion = await self.llm_services.call_openai_llm(
                    messages_for_api, model_id=model_to_use,
                    temperature=0, max_tokens=50 )
                if completion: location = completion.strip()
            
            if not location and self.llm_services.is_provider_configured("groq"): 
                logger.info("OpenAI location extraction failed or not configured, trying Groq.")
                completion = await self.llm_services.call_llm_api(
                    "Groq", self.llm_services.groq_api_key, self.llm_services.groq_endpoint_url, 
                    self.llm_services.groq_model_id, messages_for_api,
                    temperature=0, max_tokens=50 )
                if completion: location = completion.strip()
        except Exception as e:
            logger.error(f"Error during LLM location extraction: {e}", exc_info=True)

        if location and location.lower() != 'none' and location.strip() != "":
            logger.info(f"LLM successfully extracted location: '{location}'")
            return location
        logger.warning(f"LLM couldn't extract location from: '{text}'.")
        return None

    async def extract_news_topic_with_llm(self, text: str, lang_code: str, 
                                          system_prompt: str, user_prompt_template: Optional[str]) -> Optional[str]:
        logger.info(f"Requesting LLM news topic extraction from text (lang: {lang_code}): '{text}'")
        user_prompt = (user_prompt_template or "{text}").format(text=text)
        messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        topic = None
        model_to_use = config.OPENAI_CLASSIFICATION_MODEL_ID
        try:
            if self.llm_services.is_provider_configured("openai"): 
                completion = await self.llm_services.call_openai_llm(
                    messages_for_api, model_id=model_to_use,
                    temperature=0, max_tokens=50 )
                if completion: topic = completion.strip()
        except Exception as e:
            logger.error(f"Error during LLM news topic extraction: {e}", exc_info=True)

        if topic and topic.lower() != 'none' and topic.strip() != "":
            logger.info(f"LLM successfully extracted news topic: '{topic}'")
            return topic
        logger.info(f"LLM found no specific news topic in '{text}'.")
        return None