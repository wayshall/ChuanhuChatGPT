import os
from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS, ENDOFTEXT
from tiktoken.load import data_gym_to_mergeable_bpe_ranks

def set_cache_dir_and_change_mapping():
    os.environ["TIKTOKEN_CACHE_DIR"] = "/Users/way/mydev/llm/models/tiktoken_cache"
    ENCODING_CONSTRUCTORS["gpt2"] = local_gpt2

def local_gpt2():
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file="vocab.bpe",
        encoder_json_file="encoder.json",
        vocab_bpe_hash="1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        encoder_json_hash="196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    )
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        # The pattern in the original GPT-2 release is:
        # r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # This is equivalent, but executes faster:
        "pat_str": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }
