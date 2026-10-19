"""MATILDA tokenizer alias: vocabulary, normalization and special tokens are unchanged."""
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


class MatildaJevTokenizer(Qwen2Tokenizer):
    pass


MatildaJevTokenizer.register_for_auto_class("AutoTokenizer")
