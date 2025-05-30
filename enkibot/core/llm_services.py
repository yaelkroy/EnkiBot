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

# enkibot/core/llm_services.py
# ==================================================================================================
# === EnkiBot LLM Services ===
# ==================================================================================================
import logging
import httpx
import openai # Direct import for openai.AsyncOpenAI
import asyncio
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class LLMServices:
    def __init__(self, openai_api_key: Optional[str], openai_model_id: str,
                 groq_api_key: Optional[str], groq_model_id: str, groq_endpoint_url: str,
                 openrouter_api_key: Optional[str], openrouter_model_id: str, openrouter_endpoint_url: str,
                 google_ai_api_key: Optional[str], google_ai_model_id: str):
        
        print("***** LLMServices __init__ STARTING *****")
        logger.info("LLMServices __init__ STARTING")
        
        self.openai_api_key = openai_api_key
        self.openai_model_id = openai_model_id 
        self.openai_async_client: Optional[openai.AsyncOpenAI] = None
        if self.openai_api_key:
            try:
                self.openai_async_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI AsyncClient initialized successfully.")
                print("INFO: OpenAI AsyncClient initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI AsyncClient: {e}")
                print(f"ERROR: Failed to initialize OpenAI AsyncClient: {e}")
        else:
            logger.warning("OpenAI API key not provided. OpenAI calls will be disabled.")
            print("WARN: OpenAI API key not provided.")

        self.groq_api_key = groq_api_key
        self.groq_model_id = groq_model_id
        self.groq_endpoint_url = groq_endpoint_url
        if self.groq_api_key: print(f"INFO: Groq configured with key: {self.groq_api_key[:5]}...")
        else: print("WARN: Groq API key not provided.")

        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model_id = openrouter_model_id
        self.openrouter_endpoint_url = openrouter_endpoint_url
        if self.openrouter_api_key: print(f"INFO: OpenRouter configured with key: {self.openrouter_api_key[:5]}...")
        else: print("WARN: OpenRouter API key not provided.")
        
        self.google_ai_api_key = google_ai_api_key
        self.google_ai_model_id = google_ai_model_id
        if self.google_ai_api_key: print(f"INFO: Google AI configured with key: {self.google_ai_api_key[:5]}...")
        else: print("WARN: Google AI API key not provided.")
        
        print("***** LLMServices __init__ COMPLETED *****")
        logger.info("LLMServices __init__ COMPLETED")

    def is_provider_configured(self, provider_name: str) -> bool:
        if provider_name == "openai":
            return bool(self.openai_async_client and self.openai_api_key)
        elif provider_name == "groq":
            return bool(self.groq_api_key and self.groq_endpoint_url and self.groq_model_id)
        elif provider_name == "openrouter":
            return bool(self.openrouter_api_key and self.openrouter_endpoint_url and self.openrouter_model_id)
        # Add Google AI check if implemented
        return False

    async def call_openai_llm(self, messages: List[Dict[str, str]], 
                              model_id: Optional[str] = None, 
                              temperature: float = 0.7, 
                              max_tokens: int = 2000, 
                              **kwargs) -> Optional[str]:
        print(f"DEBUG: Attempting call_openai_llm. Configured: {self.is_provider_configured('openai')}")
        if not self.is_provider_configured("openai"):
            logger.warning("OpenAI client not initialized or API key missing. Cannot make call.")
            print("WARN: OpenAI client not init or key missing in call_openai_llm.")
            return None
        
        actual_model_id = model_id or self.openai_model_id
        logger.info(f"Calling OpenAI (model: {actual_model_id}) with {len(messages)} messages.")
        print(f"INFO: Calling OpenAI (model: {actual_model_id}) messages_count: {len(messages)}")
        # logger.debug(f"OpenAI messages: {messages}") # Potentially very verbose

        call_params = {
            "model": actual_model_id, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
            **kwargs 
        }
        try:
            print(f"DEBUG: Before OpenAI completions.create with params: {call_params.get('model')}, temp: {call_params.get('temperature')}")
            completion = await self.openai_async_client.chat.completions.create(**call_params)
            print(f"DEBUG: OpenAI completion object received: {type(completion)}")
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                response_content = completion.choices[0].message.content.strip()
                print(f"INFO: OpenAI successful response (first 50 chars): {response_content[:50]}")
                return response_content
            logger.warning(f"OpenAI call to {actual_model_id} returned no content or unexpected structure. Choices: {completion.choices}")
            print(f"WARN: OpenAI call to {actual_model_id} no content. Choices: {completion.choices}")
            return None
        except openai.APIStatusError as e: # More specific error type
            logger.error(f"OpenAI API Status Error (model: {actual_model_id}): HTTP Status {e.status_code} - {e.message}", exc_info=False)
            print(f"ERROR: OpenAI API Status Error (model: {actual_model_id}): HTTP Status {e.status_code} - {e.message}")
            logger.debug(f"OpenAI API Full Status Error Details: {e.response.text if e.response else 'No response body'}")
            return None
        except openai.APIError as e: 
            logger.error(f"OpenAI API Error (model: {actual_model_id}): {e.message}", exc_info=False)
            print(f"ERROR: OpenAI API Error (model: {actual_model_id}): {e.message}")
            logger.debug(f"OpenAI API Full Error Details: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI API (model: {actual_model_id}): {e}", exc_info=True)
            print(f"ERROR: Unexpected OpenAI error (model: {actual_model_id}): {e}")
            return None

    async def call_llm_api(self, provider_name: str, api_key: Optional[str], endpoint_url: Optional[str], 
                           model_id: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        print(f"DEBUG: Attempting call_llm_api for {provider_name}. Key: {'Set' if api_key else 'Not Set'}")
        if not api_key or not endpoint_url:
            logger.warning(f"{provider_name} not configured (key or URL missing). Skipping call.")
            print(f"WARN: {provider_name} not configured in call_llm_api.")
            return None
        
        logger.info(f"Calling {provider_name} ({model_id}) with {len(messages)} messages.")
        print(f"INFO: Calling {provider_name} ({model_id}) messages_count: {len(messages)}")
        # logger.debug(f"{provider_name} messages: {messages}") # Potentially verbose

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider_name.lower() == "openrouter": 
            headers.update({"HTTP-Referer": "http://localhost:8000", "X-Title": "EnkiBot"}) 

        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        
        try:
            print(f"DEBUG: Before {provider_name} POST to {endpoint_url}. Model: {model_id}")
            async with httpx.AsyncClient() as client:
                resp = await client.post(endpoint_url, json=payload, headers=headers, timeout=30.0)
            print(f"DEBUG: {provider_name} response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                response_content = data["choices"][0]["message"]["content"].strip()
                print(f"INFO: {provider_name} successful response (first 50 chars): {response_content[:50]}")
                return response_content
            logger.warning(f"{provider_name} call to {model_id} returned no content or unexpected structure. Data: {data.get('choices')}")
            print(f"WARN: {provider_name} call to {model_id} no content. Data: {data.get('choices')}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error for {provider_name} ({model_id}): {e.response.status_code} - {e.response.text}", exc_info=False)
            print(f"ERROR: HTTP Error for {provider_name} ({model_id}): {e.response.status_code} - {e.response.text}")
            logger.debug(f"{provider_name} Full Error Response Content: {e.response.content}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with {provider_name} API ({model_id}): {e}", exc_info=True)
            print(f"ERROR: Unexpected {provider_name} error ({model_id}): {e}")
            return None

    async def race_llm_calls(self, messages: List[Dict[str, str]]) -> Optional[str]:
        tasks = []
        # Store provider name with task for better logging
        task_info: List[Tuple[asyncio.Task, str]] = []

        print("DEBUG: Preparing tasks for race_llm_calls.")
        if self.is_provider_configured("openai"):
            task = asyncio.create_task(self.call_openai_llm(messages, model_id=self.openai_model_id))
            task_info.append((task, "OpenAI"))
            print("DEBUG: OpenAI task created for race.")
        if self.is_provider_configured("groq"):
            task = asyncio.create_task(self.call_llm_api("Groq", self.groq_api_key, self.groq_endpoint_url, self.groq_model_id, messages))
            task_info.append((task, "Groq"))
            print("DEBUG: Groq task created for race.")
        if self.is_provider_configured("openrouter"):
            task = asyncio.create_task(self.call_llm_api("OpenRouter", self.openrouter_api_key, self.openrouter_endpoint_url, self.openrouter_model_id, messages))
            task_info.append((task, "OpenRouter"))
            print("DEBUG: OpenRouter task created for race.")
        
        if not task_info:
            logger.warning("No LLM providers configured for racing calls.")
            print("WARN: No LLM providers for race_llm_calls.")
            return None

        logger.info(f"Racing LLM calls to: {[name for _, name in task_info]}")
        print(f"INFO: Racing LLM calls to: {[name for _, name in task_info]}")
        
        for future, provider_name in [(ti[0], ti[1]) for ti in task_info]: # Iterate through original task_info list
            # This inner loop structure with as_completed is a bit off.
            # Correct way: iterate through as_completed(just the tasks)
            # then find which provider it was. Simpler:
            pass # Will be rewritten below

        # Corrected loop for asyncio.as_completed
        tasks_only = [ti[0] for ti in task_info]
        for future in asyncio.as_completed(tasks_only):
            provider_name_for_log = "Unknown"
            # Find the provider name for the completed future
            for task_obj, name_str in task_info:
                if task_obj == future:
                    provider_name_for_log = name_str
                    break
            try:
                result = await future
                if result and result.strip():
                    logger.info(f"Successful response from {provider_name_for_log} in race.")
                    print(f"INFO: Successful response from {provider_name_for_log} in race.")
                    return result.strip()
                else:
                    # This case (result is None or empty) is already logged in individual call methods
                    logger.warning(f"{provider_name_for_log} returned no content in race.")
                    print(f"WARN: {provider_name_for_log} returned no content in race.")
            except Exception as e: 
                logger.warning(f"Provider {provider_name_for_log} failed in race_llm_calls (error should be logged by provider's method): {type(e).__name__}")
                print(f"WARN: Provider {provider_name_for_log} failed in race: {type(e).__name__}")
        
        logger.error("All LLM providers failed or returned no content in race_llm_calls.")
        print("ERROR: All LLM providers failed in race_llm_calls.")
        return None