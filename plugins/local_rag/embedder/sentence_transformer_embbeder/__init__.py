from typing import Type

from .. import AutoEmbedder, Embedder

__all__ = [
    "get_sentence_transformers_paraphrase_multilingual_minilm_l12_v2_embedder_class"
]


@AutoEmbedder.register("sentence_transformers_paraphrase_multilingual_minilm_l12_v2")
def get_sentence_transformers_paraphrase_multilingual_minilm_l12_v2_embedder_class() -> (
    Type[Embedder]
):
    from .sentence_transformers_paraphrase_multilingual_minilm_l12_v2_embedder import (
        SentenceTransformersParaphraseMultilingualMiniLML12V2Embedder,
    )

    return SentenceTransformersParaphraseMultilingualMiniLML12V2Embedder
