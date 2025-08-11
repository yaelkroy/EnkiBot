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

# -------------------------------------------------------------------------------
# Future Improvements:
# - Improve modularity to support additional features and services.
# - Enhance error handling and logging for better maintenance.
# - Expand unit tests to cover more edge cases.
# -------------------------------------------------------------------------------
# ==================================================================================================
# === EnkiBot LLM Services ===
# ==================================================================================================
# enkibot/core/llm_services.py
# (Your GPLv3 Header)

import logging
import httpx
from types import SimpleNamespace
try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover
    openai = SimpleNamespace(AsyncOpenAI=None)
import asyncio
import time
from typing import List, Dict, Optional, Any, Tuple

from enkibot import config
from enkibot.utils.provider_metrics import ProviderMetrics

logger = logging.getLogger(__name__)

class LLMServices:
    def __init__(self, openai_api_key: Optional[str], openai_model_id: str,
                 groq_api_key: Optional[str], groq_model_id: str, groq_endpoint_url: str,
                 openrouter_api_key: Optional[str], openrouter_model_id: str, openrouter_endpoint_url: str,
                 google_ai_api_key: Optional[str], google_ai_model_id: str):
        
        logger.info("LLMServices __init__ STARTING")
        
        self.openai_api_key = openai_api_key
        self.openai_model_id = openai_model_id
        self.openai_deep_research_model_id = config.OPENAI_DEEP_RESEARCH_MODEL_ID
        self.openai_embedding_model_id = config.OPENAI_EMBEDDING_MODEL_ID
        self.openai_classification_model_id = config.OPENAI_CLASSIFICATION_MODEL_ID
        self.openai_translation_model_id = config.OPENAI_TRANSLATION_MODEL_ID
        self.openai_dalle_model_id = config.OPENAI_DALLE_MODEL_ID
        self.openai_whisper_model_id = config.OPENAI_WHISPER_MODEL_ID # New

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

        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model_id = openrouter_model_id
        self.openrouter_endpoint_url = openrouter_endpoint_url

        self.google_ai_api_key = google_ai_api_key
        self.google_ai_model_id = google_ai_model_id

        # Persistent HTTP client for provider calls
        self.http_client = httpx.AsyncClient()

        # Metrics tracking structures
        self.metrics: Dict[str, ProviderMetrics] = {
            "OpenAI": ProviderMetrics(),
            "Groq": ProviderMetrics(),
            "OpenRouter": ProviderMetrics(),
        }

        self.cost_per_1k_tokens = {
            "OpenAI": config.OPENAI_COST_PER_1K_TOKENS,
            "Groq": config.GROQ_COST_PER_1K_TOKENS,
            "OpenRouter": config.OPENROUTER_COST_PER_1K_TOKENS,
        }

        logger.info("LLMServices __init__ COMPLETED")

    # ... (is_provider_configured, call_openai_llm, call_llm_api, race_llm_calls, generate_image_openai methods remain the same) ...
    def is_provider_configured(self, provider_name: str) -> bool:
        provider_name_lower = provider_name.lower()
        if provider_name_lower == "openai":
            return bool(self.openai_async_client and self.openai_api_key)
        elif provider_name_lower == "groq":
            return bool(self.groq_api_key and self.groq_endpoint_url and self.groq_model_id)
        elif provider_name_lower == "openrouter":
            return bool(self.openrouter_api_key and self.openrouter_endpoint_url and self.openrouter_model_id)
        return False

    def _record_metrics(self, provider: str, latency: float, tokens: int = 0) -> None:
        metrics = self.metrics.setdefault(provider, ProviderMetrics())
        cost_per_1k = self.cost_per_1k_tokens.get(provider, 0.0)
        metrics.record(latency, tokens, cost_per_1k)

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
        logger.debug(f"OpenAI messages: {messages}")
        call_params = { "model": actual_model_id, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs }
        try:
            start = time.perf_counter()
            completion = await self.openai_async_client.chat.completions.create(**call_params)
            latency = time.perf_counter() - start
            tokens = 0
            if getattr(completion, "usage", None):
                tokens = getattr(completion.usage, "total_tokens", 0)
            self._record_metrics("OpenAI", latency, tokens)
            logger.debug(f"OpenAI raw response: {completion}")
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                return completion.choices[0].message.content.strip()
            logger.warning(f"OpenAI call to {actual_model_id} returned no content or unexpected structure.")
            return None
        except openai.APIError as e:
            logger.error(f"OpenAI API Error (model: {actual_model_id}): {e.message}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI API (model: {actual_model_id}): {e}", exc_info=True)
        return None

    async def call_openai_deep_research(self, messages: List[Dict[str, str]],
                                        max_output_tokens: int = 1000) -> Optional[str]:
        """Call OpenAI's deep research model with web search enabled.

        This uses the Responses API so the model can issue `web_search_preview`
        tool calls when checking claims. It returns the aggregated text response
        or ``None`` if the call fails.
        """
        if not self.is_provider_configured("openai"):
            logger.warning("OpenAI client not initialized or API key missing. Cannot make deep research call.")
            return None
        try:
            start = time.perf_counter()
            response = await self.openai_async_client.responses.create(
                model=self.openai_deep_research_model_id,
                input=messages,
                tools=[{"type": "web_search_preview"}],
                tool_choice="auto",
                reasoning={"effort": "medium"},
                max_output_tokens=max_output_tokens,
            )
            latency = time.perf_counter() - start
            tokens = getattr(response, "usage", {}).get("total_tokens", 0)
            self._record_metrics("OpenAI", latency, tokens)
            return getattr(response, "output_text", None)
        except openai.APIError as e:
            logger.error(f"OpenAI API Error (deep research): {e.message}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error with OpenAI deep research: {e}", exc_info=True)
        return None

    async def embed_texts_openai(self, texts: List[str], model_id: Optional[str] = None) -> Optional[List[List[float]]]:
        """Return embeddings for ``texts`` using OpenAI's embedding endpoint.

        Falls back to ``None`` if OpenAI is not configured or an error occurs.
        """
        if not self.is_provider_configured("openai"):
            logger.warning("OpenAI client not initialized or API key missing. Cannot create embeddings.")
            return None
        actual_model_id = model_id or self.openai_embedding_model_id
        try:
            start = time.perf_counter()
            response = await self.openai_async_client.embeddings.create(model=actual_model_id, input=texts)
            latency = time.perf_counter() - start
            tokens = getattr(response, "usage", {}).get("total_tokens", 0)
            self._record_metrics("OpenAI", latency, tokens)
            if response.data:
                return [item.embedding for item in response.data]
        except openai.APIError as e:
            logger.error(f"OpenAI Embedding API Error (model: {actual_model_id}): {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI embedding call (model: {actual_model_id}): {e}", exc_info=True)
        return None

    async def call_llm_api(self, provider_name: str, api_key: Optional[str], endpoint_url: Optional[str], 
                           model_id: str, messages: List[Dict[str, str]], 
                           temperature: float = 0.7, max_tokens: int = 2000,
                           **kwargs) -> Optional[str]:
        if not api_key or not endpoint_url:
            logger.warning(f"{provider_name} not configured. Skipping call.")
            return None
        logger.info(f"Calling {provider_name} (model: {model_id}) with {len(messages)} messages.")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider_name.lower() == "openrouter":
            headers.update({"HTTP-Referer": "http://localhost:8000", "X-Title": "EnkiBot"})
        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kwargs}
        try:
            start = time.perf_counter()
            resp = await self.http_client.post(endpoint_url, json=payload, headers=headers, timeout=30.0)
            latency = time.perf_counter() - start
            resp.raise_for_status()
            data = resp.json()
            tokens = data.get("usage", {}).get("total_tokens", 0)
            self._record_metrics(provider_name, latency, tokens)
            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                return data["choices"][0]["message"]["content"].strip()
            logger.warning(f"{provider_name} call to {model_id} returned no content or unexpected structure.")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error for {provider_name} ({model_id}): {e.response.status_code} - Response: {e.response.text[:500]}...", exc_info=False)
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
                logger.warning(f"Provider {provider_name_for_log} task raised an exception during race: {type(e).__name__} - {e}")
        logger.error("All LLM providers failed or returned no content in race_llm_calls.")
        return None

    async def generate_image_openai(self, prompt: str, n: int = 1, size: str = "1024x1024", quality: str = "standard", response_format: str = "url") -> Optional[List[Dict[str, str]]]:
        if not self.is_provider_configured("openai"):
            logger.error("OpenAI client not configured. Cannot generate image.")
            return None
        model_to_use = self.openai_dalle_model_id 
        logger.info(f"Requesting DALL-E image generation with prompt: '{prompt[:70]}...' using model {model_to_use}")
        try:
            response = await self.openai_async_client.images.generate(
                model=model_to_use, prompt=prompt, n=n, size=size, quality=quality, response_format=response_format)
            image_data_list = []
            if response.data:
                for image_obj in response.data:
                    if response_format == "url" and image_obj.url: image_data_list.append({"url": image_obj.url})
                    elif response_format == "b64_json" and image_obj.b64_json: image_data_list.append({"b64_json": image_obj.b64_json})
            if image_data_list:
                logger.info(f"DALL-E generated {len(image_data_list)} image(s).")
                return image_data_list
            else:
                logger.warning(f"DALL-E call successful but no image data returned.")
                return None
        except openai.APIError as e:
            logger.error(f"OpenAI DALL-E API Error: {e.message}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error during DALL-E image generation: {e}", exc_info=True)
        return None

    async def generate_image_with_dalle(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
    ) -> Optional[List[Dict[str, str]]]:
        """Backward compatible DALL-E image generation wrapper.

        Historically the project exposed ``generate_image_with_dalle``.
        The new OpenAI client consolidates image generation under
        :meth:`generate_image_openai`.  Some parts of the codebase – and
        possibly third-party plugins – may still call the old name.  To
        prevent ``AttributeError`` exceptions we keep this thin wrapper
        that forwards the call to :meth:`generate_image_openai` and emits a
        warning to encourage migrating to the new API.
        """

        logger.warning(
            "generate_image_with_dalle is deprecated; use generate_image_openai instead."
        )
        return await self.generate_image_openai(
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            response_format=response_format,
        )

    # --- NEW METHOD FOR TRANSCRIPTION ---
    async def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribes an audio file using OpenAI's Whisper model.
        """
        if not self.is_provider_configured("openai"):
            logger.error("OpenAI client not configured. Cannot transcribe audio.")
            return None
        
        model_to_use = self.openai_whisper_model_id
        logger.info(f"Requesting Whisper transcription for file: {audio_file_path} using model {model_to_use}")

        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = await self.openai_async_client.audio.transcriptions.create(
                    model=model_to_use,
                    file=audio_file
                )
            
            transcribed_text = transcription.text
            if transcribed_text and transcribed_text.strip():
                logger.info(f"Whisper transcription successful. Text: '{transcribed_text[:100]}...'")
                return transcribed_text.strip()
            else:
                logger.warning("Whisper transcription returned empty text.")
                return None
        except openai.APIError as e:
            logger.error(f"OpenAI Whisper API Error: {e.message}", exc_info=False)
        except Exception as e:
            logger.error(f"Unexpected error during audio transcription: {e}", exc_info=True)
        return None

    # --- NEW METHOD FOR MODERATION ---
    async def moderate_text_openai(self, text: str) -> Optional[Dict[str, Any]]:
        """Calls OpenAI's moderation endpoint on the supplied text.

        Returns a dictionary with the moderation result or ``None`` if the
        moderation service is not available or an error occurred.
        """
        if not self.is_provider_configured("openai"):
            logger.warning("OpenAI client not configured. Cannot moderate text.")
            return None

        try:
            response = await self.openai_async_client.moderations.create(
                model="omni-moderation-latest",
                input=text,
            )
            if response.results:
                result = response.results[0]
                # Extract minimal useful information. ``result`` is an
                # OpenAI object; convert to a standard dict for downstream
                # processing. Include ``category_scores`` so callers can
                # compute risk levels without re-parsing the OpenAI object.
                return {
                    "flagged": bool(getattr(result, "flagged", False)),
                    "categories": getattr(result, "categories", {}),
                    "category_scores": getattr(result, "category_scores", {}),
                }
        except openai.APIError as e:
            logger.error(f"OpenAI Moderation API Error: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error during moderation call: {e}", exc_info=True)
        return None

    async def aclose(self) -> None:
        """Close underlying HTTP clients."""
        try:
            await self.http_client.aclose()
        finally:
            if self.openai_async_client:
                await self.openai_async_client.aclose()
