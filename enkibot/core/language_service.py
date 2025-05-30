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

# enkibot/core/language_service.py
import logging
import json
import os
import re
from typing import Dict, Any, Optional, List

from telegram import Update # For type hinting update_context

from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from enkibot import config # For default lang and lang_packs_dir
from enkibot.core.llm_services import LLMServices # For translation
from enkibot.utils.database import DatabaseManager # For fetching chat history

try:
    DetectorFactory.seed = 0
except Exception as e:
    logging.warning(f"Could not seed DetectorFactory for LanguageService: {e}")

logger = logging.getLogger(__name__)

class LanguageService:
    def __init__(self, 
                 llm_services: LLMServices, 
                 db_manager: DatabaseManager,
                 lang_packs_dir: str = config.LANGUAGE_PACKS_DIR, 
                 default_lang: str = config.DEFAULT_LANGUAGE):
        
        logger.info("LanguageService __init__ STARTING")
        self.llm_services = llm_services
        self.db_manager = db_manager
        self.lang_packs_dir = lang_packs_dir
        self.default_language = default_lang
        self.primary_fallback_lang = default_lang # Typically 'en'
        self.secondary_fallback_lang = "ru"

        self.language_packs: Dict[str, Dict[str, Any]] = {}
        self.llm_prompt_sets: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.response_strings: Dict[str, Dict[str, str]] = {}
        
        self.current_lang: str = self.default_language
        self.current_llm_prompt_sets: Dict[str, Dict[str, str]] = {}
        self.current_response_strings: Dict[str, str] = {}
        self.current_lang_pack_full: Dict[str, Any] = {}

        self._load_language_packs() # This will also set initial current_lang context
        logger.info("LanguageService __init__ COMPLETED")

    def _load_language_packs(self):
        self.language_packs = {}
        self.llm_prompt_sets = {}
        self.response_strings = {}
        if not os.path.exists(self.lang_packs_dir):
            logger.error(f"Language packs directory not found: {self.lang_packs_dir}")
            try:
                os.makedirs(self.lang_packs_dir, exist_ok=True)
                logger.info(f"Created language packs directory: {self.lang_packs_dir}")
            except OSError as e:
                logger.error(f"Could not create language packs directory {self.lang_packs_dir}: {e}")
        
        for lang_file in os.listdir(self.lang_packs_dir):
            if lang_file.endswith(".json"):
                lang_code = lang_file[:-5]
                file_path = os.path.join(self.lang_packs_dir, lang_file)
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f: 
                        pack_content = json.load(f)
                        self.language_packs[lang_code] = pack_content
                        self.llm_prompt_sets[lang_code] = pack_content.get("prompts", {})
                        self.response_strings[lang_code] = pack_content.get("responses", {})
                        logger.info(f"Successfully loaded language pack: {lang_code} from {file_path}")
                except json.JSONDecodeError as jde:
                    logger.error(f"Error decoding JSON from language file: {lang_file}. Error: {jde.msg} at L{jde.lineno} C{jde.colno} (char {jde.pos})", exc_info=False)
                except Exception as e:
                    logger.error(f"Error loading language file {lang_file}: {e}", exc_info=True)
        
        self._set_current_language_internals(self.default_language)


    def _set_current_language_internals(self, lang_code_to_set: str):
        chosen_lang_code = lang_code_to_set

        if chosen_lang_code not in self.language_packs:
            logger.warning(f"Language pack for initially requested '{chosen_lang_code}' not found.")
            if self.primary_fallback_lang in self.language_packs:
                logger.info(f"Falling back to primary fallback: '{self.primary_fallback_lang}'.")
                chosen_lang_code = self.primary_fallback_lang
            elif self.secondary_fallback_lang in self.language_packs:
                logger.info(f"Primary fallback '{self.primary_fallback_lang}' not found. Falling back to secondary: '{self.secondary_fallback_lang}'.")
                chosen_lang_code = self.secondary_fallback_lang
            elif self.language_packs: 
                first_available = next(iter(self.language_packs))
                logger.error(f"Fallbacks ('{self.primary_fallback_lang}', '{self.secondary_fallback_lang}') not found. Using first available: '{first_available}'.")
                chosen_lang_code = first_available
            else: 
                logger.critical("CRITICAL: No language packs loaded at all. Service may be impaired.")
                self.current_lang = "none" 
                self.current_lang_pack_full = {}; self.current_llm_prompt_sets = {}; self.current_response_strings = {}
                return

        self.current_lang = chosen_lang_code
        self.current_lang_pack_full = self.language_packs.get(chosen_lang_code, {})
        self.current_llm_prompt_sets = self.llm_prompt_sets.get(chosen_lang_code, {})
        self.current_response_strings = self.response_strings.get(chosen_lang_code, {})
        
        if not self.current_llm_prompt_sets and not self.current_response_strings:
             logger.error(f"Language '{self.current_lang}' pack loaded, but it seems empty (no 'prompts' or 'responses').")
        else:
            logger.info(f"LanguageService: Successfully set current language context to: '{self.current_lang}'")

    async def _create_and_load_language_pack(self, new_lang_code: str, update_context: Optional[Update] = None) -> bool:
        logger.info(f"LanguageService: Attempting to create language pack for new language: {new_lang_code}")
        english_pack_key = "en"
        if english_pack_key not in self.language_packs:
            logger.error(f"Cannot create new language pack: Source English ('{english_pack_key}') pack not found.")
            if update_context and update_context.effective_message:
                 await update_context.effective_message.reply_text(self.get_response_string("language_pack_creation_failed_fallback", # Use get_response_string
                                                                                           "My apologies, I'm having trouble setting up support for this language right now (missing base files)."))
            return False

        english_pack_content_str = json.dumps(self.language_packs[english_pack_key], ensure_ascii=False, indent=2)
        target_language_name = new_lang_code 

        translation_system_prompt = (
             f"You are an expert translation AI. Your task is to translate a complete JSON language pack from English to {target_language_name} (language code: {new_lang_code}).\n"
            "You MUST maintain the original JSON structure and all original keys (e.g., \"prompts\", \"responses\", \"weather_conditions_map\", \"days_of_week\", and all keys within them). Only translate the string values associated with the keys.\n"
            "The output MUST be a single, valid JSON object and nothing else. Do not add any explanatory text, comments, or markdown before or after the JSON.\n"
            "Ensure all translated strings are appropriate for a friendly AI assistant and are natural-sounding in the target language. Pay special attention to escaping characters within JSON strings if necessary (e.g. double quotes inside a string should be \\\", newlines as \\n)."
        )
        translation_user_prompt = f"Translate the following English JSON language pack to {target_language_name} ({new_lang_code}):\n\n{english_pack_content_str}"
        
        messages_for_api = [{"role": "system", "content": translation_system_prompt}, {"role": "user", "content": translation_user_prompt}]
        response_format_arg = {"response_format": {"type": "json_object"}}
        
        translated_content_str: Optional[str] = None
        try:
            # Using a specific model for translation might be good. config.OPENAI_MODEL_ID is used for general tasks.
            # Consider a dedicated translation model if available, or ensure the general one is capable.
            translator_model_id = config.OPENAI_MODEL_ID # Or a specific one like "gpt-4o-mini"
            translated_content_str = await self.llm_services.call_openai_llm(
                messages_for_api, model_id=translator_model_id, 
                temperature=0.1, max_tokens=4000, **response_format_arg
            )
        except Exception as e:
            logger.error(f"LLM call itself failed during translation for {new_lang_code}: {e}", exc_info=True)
            if update_context and update_context.effective_message:
                 await update_context.effective_message.reply_text(self.get_response_string("language_pack_creation_failed_fallback"))
            return False

        if not translated_content_str:
            logger.error(f"LLM failed to provide a translation string for {new_lang_code}.")
            if update_context and update_context.effective_message:
                 await update_context.effective_message.reply_text(self.get_response_string("language_pack_creation_failed_fallback"))
            return False
        
        clean_response = translated_content_str.strip() 
        try:
            match = re.search(r"```json\s*(.*?)\s*```", clean_response, re.DOTALL | re.IGNORECASE)
            if match: clean_response = match.group(1).strip()
            else:
                if clean_response.startswith("```json"): clean_response = clean_response[7:]
                if clean_response.endswith("```"): clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            logger.debug(f"Attempting to parse cleaned LLM translation for {new_lang_code}: '{clean_response[:300]}...'")
            translated_pack_content = json.loads(clean_response) 
            
            if not all(k in translated_pack_content for k in ["prompts", "responses", "weather_conditions_map", "days_of_week"]):
                logger.error(f"Translated pack for {new_lang_code} is missing core top-level keys. Aborting save.")
                raise ValueError("Translated JSON missing core keys.")

            new_pack_path = os.path.join(self.lang_packs_dir, f"{new_lang_code}.json")
            with open(new_pack_path, 'w', encoding='utf-8') as f: 
                json.dump(translated_pack_content, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully created and saved new language pack: {new_lang_code}.json")
            
            self.language_packs[new_lang_code] = translated_pack_content
            self.llm_prompt_sets[new_lang_code] = translated_pack_content.get("prompts", {})
            self.response_strings[new_lang_code] = translated_pack_content.get("responses", {})
            logger.info(f"New language pack for {new_lang_code} is now available at runtime.")
            return True
        except json.JSONDecodeError as jde:
            logger.error(
                f"Failed to decode LLM translation JSON for {new_lang_code}. Error: {jde.msg} "
                f"at L{jde.lineno} C{jde.colno} (char {jde.pos}). "
                f"Nearby: '{clean_response[max(0, jde.pos-50):jde.pos+50]}'", 
                exc_info=False 
            )
            log_limit = 3000
            full_resp_to_log = translated_content_str 
            if len(full_resp_to_log) < log_limit: logger.debug(f"Full problematic translated content for {new_lang_code}:\n{full_resp_to_log}")
            else: logger.debug(f"Problematic translated content (first {log_limit} chars) for {new_lang_code}:\n{full_resp_to_log[:log_limit]}")
            if update_context and update_context.effective_message:
                await update_context.effective_message.reply_text(self.get_response_string("language_pack_creation_failed_fallback"))
            return False
        except Exception as e:
            logger.error(f"Error processing/saving new lang pack for {new_lang_code}: {e}", exc_info=True)
            if update_context and update_context.effective_message: 
                await update_context.effective_message.reply_text(self.get_response_string("language_pack_creation_failed_fallback"))
            return False

    async def determine_language_context(self, 
                                         current_message_text: Optional[str], 
                                         chat_id: Optional[int], 
                                         update_context: Optional[Update] = None) -> str:
        """
        Determines language using current message and chat history, sets context,
        and attempts to create new language packs if needed.
        """
        CONFIDENCE_THRESHOLD = 0.60
        NUM_RECENT_MESSAGES_FOR_DETECTION = 3
        MIN_AGGREGATED_TEXT_LENGTH = 20

        candidate_lang_code = self.current_lang if self.current_lang != "none" else self.default_language
        
        texts_for_detection: List[str] = []
        if current_message_text and current_message_text.strip():
            texts_for_detection.append(current_message_text.strip())

        if chat_id and (not texts_for_detection or len(texts_for_detection[0]) < 50): 
            logger.debug(f"Fetching last {NUM_RECENT_MESSAGES_FOR_DETECTION} messages from chat {chat_id} for lang detection.")
            try:
                if self.db_manager:
                    recent_messages = await self.db_manager.get_recent_chat_texts(chat_id, limit=NUM_RECENT_MESSAGES_FOR_DETECTION)
                    if recent_messages:
                        texts_for_detection.extend(recent_messages) 
                        logger.debug(f"Fetched {len(recent_messages)} messages. Total texts for detection: {len(texts_for_detection)}")
                else: logger.warning("db_manager not available in determine_language_context.")
            except Exception as e: logger.error(f"Error fetching recent chat texts: {e}", exc_info=True)
        
        aggregated_text = " . ".join(reversed(texts_for_detection)).strip()
        logger.debug(f"Aggregated text for lang detection (len {len(aggregated_text)}): '{aggregated_text[:100]}...'")

        detected_languages_with_probs: List[Any] = []
        attempted_new_detection_on_aggregated = False

        if aggregated_text and len(aggregated_text) >= MIN_AGGREGATED_TEXT_LENGTH:
            attempted_new_detection_on_aggregated = True
            try:
                detected_languages_with_probs = detect_langs(aggregated_text)
                logger.info(f"LangDetect results on aggregated text: {detected_languages_with_probs}")
            except Exception as e: 
                logger.warning(f"LangDetect error on aggregated text: {e}. Current candidate: '{candidate_lang_code}'.")
        else:
            logger.info(f"Aggregated text too short. Using current candidate: '{candidate_lang_code}'.")

        chosen_lang_from_detection = None
        if detected_languages_with_probs:
            for lang_obj in sorted(detected_languages_with_probs, key=lambda x: x.prob, reverse=True):
                lang_code_detected, lang_prob = lang_obj.lang, lang_obj.prob
                logger.debug(f"Checking detected lang: {lang_code_detected} with prob: {lang_prob:.3f}")
                if lang_prob >= CONFIDENCE_THRESHOLD:
                    if lang_code_detected in self.language_packs:
                        logger.info(f"Using existing pack for detected '{lang_code_detected}' (prob: {lang_prob:.3f}).")
                        chosen_lang_from_detection = lang_code_detected
                        break 
                    elif chosen_lang_from_detection is None: 
                        chosen_lang_from_detection = lang_code_detected
                        logger.info(f"High-confidence lang '{lang_code_detected}' (prob {lang_prob:.3f}) is candidate for creation.")
                else: break 
        
        if chosen_lang_from_detection:
            candidate_lang_code = chosen_lang_from_detection
            logger.info(f"Candidate language after detection: '{candidate_lang_code}'")
        else:
            logger.info(f"No suitable language from detection. Sticking with candidate: '{candidate_lang_code}'")

        if candidate_lang_code not in self.language_packs:
            if attempted_new_detection_on_aggregated or candidate_lang_code != self.current_lang :
                logger.warning(f"Language '{candidate_lang_code}' pack not found. Attempting creation.")
                if await self._create_and_load_language_pack(candidate_lang_code, update_context=update_context):
                    self._set_current_language_internals(candidate_lang_code)
                else:
                    logger.warning(f"Failed to create pack for '{candidate_lang_code}'. Applying fallbacks.")
                    self._set_current_language_internals(self.default_language)
            else:
                logger.info(f"Pack for current/default candidate '{candidate_lang_code}' missing. Applying fallbacks.")
                self._set_current_language_internals(self.default_language)
        else: 
            self._set_current_language_internals(candidate_lang_code)
        
        return self.current_lang

    def get_llm_prompt_set(self, key: str) -> Optional[Dict[str, str]]:
        current_prompts_to_check = self.current_llm_prompt_sets
        prompt_set = current_prompts_to_check.get(key)
        
        if not prompt_set: 
            logger.debug(f"LLM prompt set key '{key}' not in '{self.current_lang}'. Trying '{self.primary_fallback_lang}'.")
            current_prompts_to_check = self.llm_prompt_sets.get(self.primary_fallback_lang, {})
            prompt_set = current_prompts_to_check.get(key)
            if not prompt_set and self.primary_fallback_lang != self.secondary_fallback_lang: 
                 logger.debug(f"LLM prompt set key '{key}' not in '{self.primary_fallback_lang}'. Trying '{self.secondary_fallback_lang}'.")
                 current_prompts_to_check = self.llm_prompt_sets.get(self.secondary_fallback_lang, {})
                 prompt_set = current_prompts_to_check.get(key)
        
        if not prompt_set:
            logger.error(f"LLM prompt set for key '{key}' ultimately not found.")
            return None
        if not isinstance(prompt_set, dict) or "system" not in prompt_set: 
            logger.error(f"LLM prompt set for key '{key}' (found) is malformed: {prompt_set}")
            return None
        return prompt_set

    def get_response_string(self, key: str, default_value: Optional[str] = None, **kwargs) -> str:
        raw_string = self.current_response_strings.get(key)
        lang_tried = self.current_lang

        if raw_string is None: 
            lang_tried = self.primary_fallback_lang
            raw_string = self.response_strings.get(self.primary_fallback_lang, {}).get(key)
            if raw_string is None and self.primary_fallback_lang != self.secondary_fallback_lang: 
                lang_tried = self.secondary_fallback_lang
                raw_string = self.response_strings.get(self.secondary_fallback_lang, {}).get(key)

        if raw_string is None: 
            if default_value is not None: raw_string = default_value
            else: logger.error(f"Response string for key '{key}' ultimately not found. Using placeholder."); raw_string = f"[[Missing response: {key}]]"
        
        try:
            return raw_string.format(**kwargs) if kwargs else raw_string
        except KeyError as e:
            logger.error(f"Missing format key '{e}' in response for '{key}' (lang tried: {lang_tried}, raw: '{raw_string}')")
            english_raw = self.response_strings.get("en", {}).get(key, f"[[Format error & missing English for key: {key}]]")
            try: return english_raw.format(**kwargs) if kwargs else english_raw
            except KeyError: return f"[[Formatting error for response key: {key} - check placeholders]]"