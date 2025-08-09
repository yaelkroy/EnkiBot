# enkibot/local_telegram_bot.py
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
"""Minimal Telegram wiring for the local two‑tier model setup.

This script bypasses the project's main application stack to present a compact
example of running EnkiBot purely with local models.  It exposes three
commands:

``/fast`` – use the 7–8B model directly.
``/deep`` – force the 70B/72B model.
``/web``  – perform a duckduckgo search, fetch top pages and summarise with
            citations via the fast model.

Any other message goes through :class:`~enkibot.modules.model_router.ModelRouter`
which escalates to the deep model when needed.
"""
from __future__ import annotations

import logging
import os

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from .modules.local_model_manager import LocalModelManager, ModelConfig
from .modules.model_router import ModelRouter
from .modules.web_tool import web_research

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAST_MODEL_PATH = os.getenv("ENKIBOT_FAST_MODEL", "mistral-7b-instruct.Q5_K_M.gguf")
DEEP_MODEL_PATH = os.getenv("ENKIBOT_DEEP_MODEL", "llama-3-70b-instruct.Q4_K_M.gguf")

manager = LocalModelManager(
    ModelConfig(FAST_MODEL_PATH, n_ctx=4096, n_threads=16),
    ModelConfig(DEEP_MODEL_PATH, n_ctx=8192, n_threads=16),
)
router = ModelRouter(manager)


async def fast_cmd(update, context):  # /fast
    prompt = " ".join(context.args)
    await update.message.reply_text(manager.generate(prompt, model="fast"))


async def deep_cmd(update, context):  # /deep
    prompt = " ".join(context.args)
    await update.message.reply_text(manager.generate(prompt, model="deep"))


async def web_cmd(update, context):  # /web query
    query = " ".join(context.args)
    docs = web_research(query, k=3)
    context_block = "\n\n".join(
        f"[{i+1}] {d['title']}\n{d['text'][:500]}" for i, d in enumerate(docs)
    )
    prompt = (
        f"Use the following web results to answer the question. Cite sources "
        f"as [number](url).\n\n{context_block}\n\nQuestion: {query}"
    )
    answer = manager.generate(prompt, model="fast")
    await update.message.reply_text(answer)


async def default_handler(update, context):
    response = router.generate(update.message.text)
    await update.message.reply_text(response)


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("fast", fast_cmd))
    app.add_handler(CommandHandler("deep", deep_cmd))
    app.add_handler(CommandHandler("web", web_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, default_handler))
    logger.info("Starting local EnkiBot")
    app.run_polling()


if __name__ == "__main__":
    main()
