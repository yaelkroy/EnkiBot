# enkibot/modules/local_model_manager.py
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
"""Thin wrapper around :mod:`llama_cpp` for loading local models.

The project historically relied on hosted LLM providers.  For users who want
fully local inference we provide :class:`LocalModelManager` which manages two
GGUF models – a *fast* one (7B/8B) and a *deep* one (70B/72B).  Both models are
loaded lazily to keep start‑up time low and to conserve memory.

Example
-------
>>> from enkibot.modules.local_model_manager import LocalModelManager
>>> manager = LocalModelManager(
...     fast_model_path="mistral-7b-instruct.Q5_K_M.gguf",
...     deep_model_path="llama-3-70b-instruct.Q4_K_M.gguf",
... )
>>> text = manager.generate("Write a haiku about modular bots", model="fast")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

try:  # llama_cpp is an optional dependency
    from llama_cpp import Llama
except Exception:  # pragma: no cover - library not always available in CI
    Llama = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single local model."""

    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8


class LocalModelManager:
    """Load and run local GGUF models via :mod:`llama_cpp`.

    Parameters
    ----------
    fast_model_cfg:
        Configuration for the smaller, faster model (e.g. Mistral‑7B or
        Llama‑3‑8B).
    deep_model_cfg:
        Configuration for the larger, higher quality model (e.g. Llama‑3‑70B or
        Qwen‑2‑72B).
    """

    def __init__(self, fast_model_cfg: ModelConfig, deep_model_cfg: ModelConfig):
        self.fast_cfg = fast_model_cfg
        self.deep_cfg = deep_model_cfg
        self._fast: Optional[Llama] = None
        self._deep: Optional[Llama] = None

    # ------------------------------------------------------------------
    # Loading helpers
    def _load_fast(self) -> Llama:
        if self._fast is None:
            if Llama is None:
                raise RuntimeError("llama_cpp is not installed")
            logger.info("Loading fast model from %s", self.fast_cfg.model_path)
            self._fast = Llama(
                model_path=self.fast_cfg.model_path,
                n_ctx=self.fast_cfg.n_ctx,
                n_threads=self.fast_cfg.n_threads,
            )
        return self._fast

    def _load_deep(self) -> Llama:
        if self._deep is None:
            if Llama is None:
                raise RuntimeError("llama_cpp is not installed")
            logger.info("Loading deep model from %s", self.deep_cfg.model_path)
            self._deep = Llama(
                model_path=self.deep_cfg.model_path,
                n_ctx=self.deep_cfg.n_ctx,
                n_threads=self.deep_cfg.n_threads,
            )
        return self._deep

    # ------------------------------------------------------------------
    # Public API
    def generate(self, prompt: str, model: str = "fast", max_tokens: int = 512,
                 temperature: float = 0.7) -> str:
        """Generate text using the requested model.

        Parameters
        ----------
        prompt:
            User prompt.
        model:
            ``"fast"`` or ``"deep"``.
        max_tokens, temperature:
            Passed to :class:`llama_cpp.Llama`.
        """

        llm = self._load_fast() if model == "fast" else self._load_deep()
        template = f"<<SYS>>You are a helpful assistant.<</SYS>>\n[INST]{prompt}[/INST]"
        result = llm(
            template,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>"],
        )
        return result["choices"][0]["text"]
