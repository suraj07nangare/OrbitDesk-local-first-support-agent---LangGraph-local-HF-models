import time
from typing import Dict, List

import numpy as np

from . import config
from .knowledge_base import Chunk, load_all_chunks
from .logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        revision: str = config.EMBEDDING_MODEL_REVISION,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        start = time.perf_counter()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision)
        self.load_time_seconds = time.perf_counter() - start
        logger.info("Loaded embedding model %s in %.2fs", model_name, self.load_time_seconds)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class RetrievalIndex:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = load_all_chunks()
        texts = [chunk.text for chunk in self.chunks]
        self.vectors = embedding_model.encode(texts)
        logger.info("Indexed %d passages", len(self.chunks))

    def search(self, query: str, top_k: int = config.TOP_K) -> List[Dict]:
        query_vector = self.embedding_model.encode([query])[0]
        scores = self.vectors @ query_vector
        ranked_indices = np.argsort(-scores)[:top_k]

        results: List[Dict] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "source_id": chunk.source_id,
                    "source_type": chunk.source_type,
                    "title": chunk.title,
                    "section": chunk.section,
                    "status": chunk.status,
                    "text": chunk.text,
                    "score": float(scores[idx]),
                }
            )
        return results
