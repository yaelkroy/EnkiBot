# enkibot/modules/rag_service.py
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
"""Very small Retrieval-Augmented Generation helper using FAISS."""

from __future__ import annotations

from typing import List, Tuple, Optional
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from enkibot import config
from enkibot.core.llm_services import LLMServices

logger = logging.getLogger(__name__)


class RAGService:
    """Vector search over a set of documents with optional OpenAI embeddings."""

    def __init__(
        self,
        llm_services: Optional[LLMServices] = None,
        embedding_model: Optional[str] = None,
    ):
        self.llm_services = llm_services
        self.use_openai = bool(
            llm_services and llm_services.is_provider_configured("openai")
        )
        self.embedding_model = (
            embedding_model
            or (
                config.OPENAI_EMBEDDING_MODEL_ID
                if self.use_openai
                else "intfloat/e5-small-v2"
            )
        )
        self.encoder: Optional[SentenceTransformer] = None
        if not self.use_openai:
            self.encoder = SentenceTransformer(self.embedding_model)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[str] = []

    async def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    async def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self.use_openai and self.llm_services:
            return await self.llm_services.embed_texts_openai(
                texts, model_id=self.embedding_model
            )
        if self.encoder:
            return self.encoder.encode(texts).tolist()
        return None

    async def add_documents(self, docs: List[str]) -> None:
        """Add raw documents to the index."""
        embeddings = await self._embed(docs)
        if not embeddings:
            return
        await self._ensure_index(len(embeddings[0]))
        self.index.add(np.array(embeddings, dtype="float32"))
        self.documents.extend(docs)
        logger.info("Indexed %d documents", len(docs))

    async def query(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top ``top_k`` documents most relevant to ``question``."""
        if not self.documents:
            return []
        emb = await self._embed([question])
        if not emb or self.index is None:
            return []
        scores, ids = self.index.search(np.array(emb, dtype="float32"), top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

