# enkibot/modules/profile_manager.py
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
# === EnkiBot Profile Manager ===
# ==================================================================================================
# Manages user psychological profiles and name variations using LLMs and database interaction.
# ==================================================================================================
import logging
import json
from typing import Optional, Dict, TYPE_CHECKING

from enkibot.utils.database import DatabaseManager 

if TYPE_CHECKING:
    from enkibot.core.llm_services import LLMServices # For type hinting

logger = logging.getLogger(__name__)

class ProfileManager:
    def __init__(self, llm_services: 'LLMServices', db_manager: DatabaseManager):
        self.llm_services = llm_services # Type hint is string literal
        self.db_manager = db_manager
        self.MAX_PROFILE_SIZE = 4000

    async def populate_name_variations_with_llm(self, user_id: int, first_name: Optional[str], 
                                                last_name: Optional[str], username: Optional[str],
                                                system_prompt: str, user_prompt_template: str):
        if not self.llm_services.is_provider_configured("openai"): # Check specific provider
            logger.warning(f"Name variation for user {user_id} skipped: OpenAI not configured in LLMServices.")
            return

        name_parts = [part for part in [first_name, last_name, username] if part and str(part).strip()]
        if not name_parts:
            logger.info(f"No valid name parts for user {user_id}, skipping name variation.")
            return
            
        name_info = ", ".join(name_parts)
        logger.info(f"Requesting name variations for user {user_id} ({name_info}).")
        user_prompt = user_prompt_template.format(name_info=name_info)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        name_variations = set([p.lower() for p in name_parts])
        
        try:
            completion_str = await self.llm_services.call_openai_llm(
                messages, temperature=0.3, response_format={"type": "json_object"}
            ) # Removed model_id, should be handled by call_openai_llm default
            if completion_str:
                data = json.loads(completion_str)
                variations_list = data.get('variations')
                if isinstance(variations_list, list):
                    name_variations.update([str(v).lower().strip() for v in variations_list if v and str(v).strip()])
                    logger.info(f"LLM got {len(variations_list)} raw variations. Total unique: {len(name_variations)} for user {user_id}.")
                else:
                    logger.warning(f"LLM name variations for {user_id} no 'variations' list. Resp: {completion_str[:200]}")
            else:
                logger.warning(f"LLM no content for name variations for user {user_id}.")
        except Exception as e:
            logger.error(f"LLM name variation error (user {user_id}): {e}", exc_info=True)

        if name_variations:
            await self.db_manager.save_user_name_variations(user_id, list(name_variations))
        else:
            logger.info(f"No name variations to save for user {user_id}.")

    async def analyze_and_update_user_profile(self, user_id: int, message_text: str,
                                              create_system_prompt: str, create_user_prompt_template: str,
                                              update_system_prompt: str, update_user_prompt_template: str):
        if not self.llm_services.is_provider_configured("openai"):
            logger.warning(f"Profiling for {user_id} skipped: OpenAI not configured.")
            return
        logger.info(f"Starting/Updating profile analysis for user {user_id}...")
        current_notes = await self.db_manager.get_user_profile_notes(user_id)
        sys_prompt, user_prompt = "", ""

        if not current_notes:
            logger.info(f"No profile for {user_id}. Creating new.")
            sys_prompt, user_prompt = create_system_prompt, create_user_prompt_template.format(message_text=message_text)
        else:
            logger.info(f"Existing profile for {user_id}. Updating.")
            sys_prompt, user_prompt = update_system_prompt, update_user_prompt_template.format(
                current_profile_notes=current_notes, message_text=message_text)
        
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        updated_notes_str: Optional[str] = None
        try:
            updated_notes_str = await self.llm_services.call_openai_llm(messages, temperature=0.5, max_tokens=1000)
        except Exception as e:
            logger.error(f"LLM profile analysis error for {user_id}: {e}", exc_info=True)

        if updated_notes_str and updated_notes_str.strip():
            await self.db_manager.update_user_profile_notes(user_id, updated_notes_str.strip()[:self.MAX_PROFILE_SIZE])
            logger.info(f"Profile updated for {user_id}.")
        else:
            logger.warning(f"Profile analysis no content for {user_id}.")
