# enkibot/modules/model_router.py
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
"""Routing logic for selecting between fast and deep local models."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from .local_model_manager import LocalModelManager

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    model: str
    clean_prompt: str
    use_web: bool = False


class ModelRouter:
    """Simple heuristic router for two‑tier local models."""

    def __init__(self, manager: LocalModelManager):
        self.manager = manager

    # ------------------------------------------------------------------
    def _needs_deep_model(self, prompt: str) -> bool:
        if prompt.startswith("/deep"):
            return True
        if len(prompt) > 400 or len(prompt.split()) > 120:
            return True
        return False

    def _needs_web(self, prompt: str) -> bool:
        triggers = ["http://", "https://", "www", "today", "current", "latest"]
        lowered = prompt.lower()
        return any(t in lowered for t in triggers) or prompt.startswith("/web")

    def route(self, prompt: str) -> RoutingDecision:
        use_web = self._needs_web(prompt)
        model = "deep" if self._needs_deep_model(prompt) else "fast"
        clean = re.sub(r"^/(deep|fast|web)\s*", "", prompt).strip()
        logger.debug("Routing prompt to %s model (web=%s)", model, use_web)
        return RoutingDecision(model=model, clean_prompt=clean, use_web=use_web)

    # ------------------------------------------------------------------
    def generate(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper that performs routing and generation."""
        decision = self.route(prompt)
        return self.manager.generate(decision.clean_prompt, model=decision.model, **kwargs)
