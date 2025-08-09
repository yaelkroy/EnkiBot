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
"""Very small Retrieval‑Augmented Generation helper using FAISS."""
from __future__ import annotations

from typing import List, Tuple
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGService:
    """CPU friendly vector search over a set of documents."""

    def __init__(self, embedding_model: str = "intfloat/e5-small-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]) -> None:
        """Add raw documents to the index."""
        embeddings = self.encoder.encode(docs)
        self.index.add(np.array(embeddings, dtype="float32"))
        self.documents.extend(docs)
        logger.info("Indexed %d documents", len(docs))

    def query(self, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top ``top_k`` documents most relevant to ``question``."""
        if not self.documents:
            return []
        emb = self.encoder.encode([question])
        scores, ids = self.index.search(np.array(emb, dtype="float32"), top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results
