from numpy import ndarray

from .. import Document, Embedder, EmbedderConfig

__all__ = ["SentenceTransformersParaphraseMultilingualMiniLML12V2Embedder"]


class SentenceTransformersParaphraseMultilingualMiniLML12V2Embedder(Embedder):
    def __init__(self, config: EmbedderConfig) -> None:
        self._config = config
        self._model = self._load_model(config)

    @staticmethod
    def _load_model(config: EmbedderConfig):
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if config.endpoint is not None:
            import os

            old_endpoint = os.environ.get("HF_ENDPOINT")
            os.environ["HF_ENDPOINT"] = config.endpoint

            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, cache_folder=config.cache_folder)
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            else:
                del os.environ["HF_ENDPOINT"]
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, cache_folder=config.cache_folder)
        if config.max_seq_length is not None:
            model.max_seq_length = min(model.max_seq_length, config.max_seq_length)
        return model

    def encode_text(self, text: str | list[str]) -> ndarray:
        return self._model.encode(text)

    def create_embeddings(self, documents: list[Document]) -> list[Document]:
        parts = []
        for document in documents:
            parts.extend(document.parts)

        batch_size = 1 if self._config.batch_size is None else self._config.batch_size
        embeddings = self._model.encode(
            [part.get_data() for part in parts], batch_size=batch_size
        )
        for i, part in enumerate(parts):
            part.embedding = embeddings[i]
        return documents

    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
