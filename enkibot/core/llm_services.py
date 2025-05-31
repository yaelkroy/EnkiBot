# enkibot/core/llm_services.py
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
# === EnkiBot LLM Services ===
# ==================================================================================================


import logging
import httpx
import openai 
import asyncio
from typing import List, Dict, Optional, Any, Tuple

from enkibot import config

logger = logging.getLogger(__name__)

class LLMServices:
    def __init__(self, openai_api_key: Optional[str], openai_model_id: str,
                 groq_api_key: Optional[str], groq_model_id: str, groq_endpoint_url: str,
                 openrouter_api_key: Optional[str], openrouter_model_id: str, openrouter_endpoint_url: str,
                 google_ai_api_key: Optional[str], google_ai_model_id: str):
        
        logger.info("***** LLMServices __init__ STARTING *****")
        
        self.openai_api_key = openai_api_key
        self.openai_model_id = openai_model_id 
        self.openai_classification_model_id = config.OPENAI_CLASSIFICATION_MODEL_ID
        self.openai_translation_model_id = config.OPENAI_TRANSLATION_MODEL_ID
        # For image generation models
        self.openai_dalle_model_id = config.OPENAI_DALLE_MODEL_ID
        self.openai_multimodal_image_model_id = config.OPENAI_MULTIMODAL_IMAGE_MODEL_ID


        self.openai_async_client: Optional[openai.AsyncOpenAI] = None
        if self.openai_api_key:
            try:
                self.openai_async_client = openai.AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI AsyncClient initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI AsyncClient: {e}")
        else:
            logger.warning("OpenAI API key not provided. OpenAI calls will be disabled.")

        self.groq_api_key = groq_api_key
        self.groq_model_id = groq_model_id
        self.groq_endpoint_url = groq_endpoint_url
        if self.groq_api_key: logger.info(f"Groq configured with key: {self.groq_api_key[:5]}...")
        else: logger.warning("Groq API key not provided.")

        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model_id = openrouter_model_id
        self.openrouter_endpoint_url = openrouter_endpoint_url
        if self.openrouter_api_key: logger.info(f"OpenRouter configured with key: {self.openrouter_api_key[:5]}...")
        else: logger.warning("OpenRouter API key not provided.")
        
        self.google_ai_api_key = google_ai_api_key
        self.google_ai_model_id = google_ai_model_id
        if self.google_ai_api_key: logger.info(f"Google AI configured with key: {self.google_ai_api_key[:5]}...")
        else: logger.warning("Google AI API key not provided.")
        
        logger.info("***** LLMServices __init__ COMPLETED *****")

    def is_provider_configured(self, provider_name: str) -> bool:
        provider_name_lower = provider_name.lower()
        if provider_name_lower == "openai":
            return bool(self.openai_async_client and self.openai_api_key)
        elif provider_name_lower == "groq":
            return bool(self.groq_api_key and self.groq_endpoint_url and self.groq_model_id)
        elif provider_name_lower == "openrouter":
            return bool(self.openrouter_api_key and self.openrouter_endpoint_url and self.openrouter_model_id)
        return False

    async def call_openai_llm(self, messages: List[Dict[str, str]], 
                              model_id: Optional[str] = None,
                              temperature: float = 0.7, 
                              max_tokens: int = 2000, 
                              **kwargs) -> Optional[str]:
        if not self.is_provider_configured("openai"):
            logger.warning("OpenAI client not initialized or API key missing. Cannot make call.")
            return None
        
        actual_model_id = model_id or self.openai_model_id 
        logger.info(f"Calling OpenAI (model: {actual_model_id}) with {len(messages)} messages.")

        call_params = { "model": actual_model_id, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs }
        try:
            logger.debug(f"Before OpenAI completions.create with params: model='{call_params.get('model')}', temp: {call_params.get('temperature')}")
            completion = await self.openai_async_client.chat.completions.create(**call_params)
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                response_content = completion.choices[0].message.content.strip()
                logger.info(f"OpenAI successful response (first 50 chars): {response_content[:50]}")
                return response_content
            logger.warning(f"OpenAI call to {actual_model_id} returned no content. Choices: {completion.choices}")
            return None
        except openai.APIStatusError as e: 
            logger.error(f"OpenAI API Status Error (model: {actual_model_id}): HTTP Status {e.status_code if hasattr(e, 'status_code') else 'N/A'} - {e.message if hasattr(e, 'message') else str(e)}", exc_info=False)
            return None
        except openai.APIError as e: 
            logger.error(f"OpenAI API Error (model: {actual_model_id}): {e.message if hasattr(e, 'message') else str(e)}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI API (model: {actual_model_id}): {e}", exc_info=True)
            return None

    async def generate_image_with_dalle(self, 
                                        prompt: str,
                                        n: int = 1,
                                        size: str = "1024x1024",
                                        quality: str = "standard",
                                        response_format: str = "url"
                                        ) -> Optional[List[Dict[str, Any]]]:
        if not self.is_provider_configured("openai"):
            logger.error("OpenAI client not configured. Cannot generate image with DALL-E.")
            return None

        model_to_use = self.openai_dalle_model_id 
        logger.info(f"Requesting image generation via DALL-E API with prompt: '{prompt[:70]}...' using model {model_to_use}")

        try:
            response = await self.openai_async_client.images.generate(
                model=model_to_use,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                response_format=response_format
            )
            
            image_data_list = []
            if response.data:
                for image_object in response.data:
                    if response_format == "url" and image_object.url:
                        image_data_list.append({"url": image_object.url})
                        logger.info(f"DALL-E generated image URL: {image_object.url}")
                    elif response_format == "b64_json" and image_object.b64_json:
                        image_data_list.append({"b64_json": image_object.b64_json})
            
            return image_data_list if image_data_list else None
        except openai.APIError as e:
            logger.error(f"OpenAI DALL-E API Error (model: {model_to_use}): Status {e.status_code if hasattr(e, 'status_code') else 'N/A'} - {e.message if hasattr(e, 'message') else str(e)}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error during DALL-E image generation: {e}", exc_info=True)
        return None

    async def generate_image_with_responses_api(self, prompt: str) -> Optional[List[Dict[str, str]]]:
        if not self.is_provider_configured("openai"):
            logger.error("OpenAI client not configured. Cannot generate image via Responses API.")
            return None

        model_to_use = self.openai_multimodal_image_model_id
        logger.info(f"Requesting image generation via Responses API with prompt: '{prompt[:70]}...' using model {model_to_use}")

        try:
            response = await self.openai_async_client.responses.create(
                model=model_to_use,
                input=prompt,
                tools=[{"type": "image_generation"}] 
            )
            image_data_list = []
            if response.output:
                for output_item in response.output:
                    if output_item.type == "image_generation_call" and output_item.result:
                        image_data_list.append({"b64_json": output_item.result})
                        logger.info(f"Image generated via Responses API (base64, first 50 chars): {output_item.result[:50]}...")
            return image_data_list if image_data_list else None
        except openai.APIError as e:
            logger.error(f"OpenAI Responses API Error (model: {model_to_use}, image gen): Status {e.status_code if hasattr(e, 'status_code') else 'N/A'} - {e.message if hasattr(e, 'message') else str(e)}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error during Responses API image generation: {e}", exc_info=True)
        return None

    async def call_llm_api(self, provider_name: str, api_key: Optional[str], endpoint_url: Optional[str], 
                            model_id: str, messages: List[Dict[str, str]], 
                            temperature: float = 0.7, max_tokens: int = 2000,
                            **kwargs 
                            ) -> Optional[str]:
        if not api_key or not endpoint_url:
            logger.warning(f"{provider_name} not configured (key or URL missing). Skipping call.")
            return None
        
        logger.info(f"Calling {provider_name} (model: {model_id}) with {len(messages)} messages.")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider_name.lower() == "openrouter": 
            headers.update({"HTTP-Referer": "http://localhost:8000", "X-Title": "EnkiBot"})

        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kwargs}
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(endpoint_url, json=payload, headers=headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                response_content = data["choices"][0]["message"]["content"].strip()
                logger.info(f"{provider_name} successful response (first 50 chars): {response_content[:50]}")
                return response_content
            logger.warning(f"{provider_name} call to {model_id} returned no content. Data: {data.get('choices')}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error for {provider_name} ({model_id}): {e.response.status_code} - Response: {e.response.text[:500]}...", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error with {provider_name} API ({model_id}): {e}", exc_info=True)
            return None

    async def race_llm_calls(self, messages: List[Dict[str, str]]) -> Optional[str]:
        task_info: List[Tuple[asyncio.Task, str]] = []

        if self.is_provider_configured("openai"):
            task = asyncio.create_task(self.call_openai_llm(messages, model_id=self.openai_model_id))
            task_info.append((task, "OpenAI"))
        if self.is_provider_configured("groq"):
            task = asyncio.create_task(self.call_llm_api("Groq", self.groq_api_key, self.groq_endpoint_url, self.groq_model_id, messages))
            task_info.append((task, "Groq"))
        if self.is_provider_configured("openrouter"):
            task = asyncio.create_task(self.call_llm_api("OpenRouter", self.openrouter_api_key, self.openrouter_endpoint_url, self.openrouter_model_id, messages))
            task_info.append((task, "OpenRouter"))
        
        if not task_info:
            logger.warning("No LLM providers configured for racing calls.")
            return None

        logger.info(f"Racing LLM calls to: {[name for _, name in task_info]}")
        
        task_to_provider_map = {task_obj: name for task_obj, name in task_info}
        tasks_only = [task_obj for task_obj, _ in task_info]

        for future in asyncio.as_completed(tasks_only):
            provider_name_for_log = task_to_provider_map.get(future, "UnknownProvider") 
            try:
                result = await future
                if result and result.strip():
                    logger.info(f"Successful response from {provider_name_for_log} in race.")
                    return result.strip()
                else:
                    logger.warning(f"{provider_name_for_log} returned no content in race.")
            except Exception as e: 
                logger.warning(f"Provider {provider_name_for_log} task failed in race: {type(e).__name__} - {e}")
        
        logger.error("All LLM providers failed or returned no content in race_llm_calls.")
        return None