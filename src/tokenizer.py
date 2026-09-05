"""Convert text to lowercase tokens, then optionally normalize them with Porter.

Tokenization and stemming are separate, visible steps. Documents and queries
must call the same normalization function with the same mode.
"""
import re

from nltk.stem import PorterStemmer

# Explicitly select NLTK's default Porter variant; no corpus data is needed.
PORTER_STEMMER = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)


def tokenize(text: str) -> list[str]:
    """Lowercase and split on punctuation and whitespace; retain letters/digits."""
    return re.findall(r"[^\W_]+", text.lower())


def normalize_tokens(tokens: list[str], stemming: bool = False) -> list[str]:
    """Return exact tokens or Porter stems without changing the input list.

    Tokens are already lowercase. Preserve occurrences: term frequency must
    count every occurrence, including different forms that share a stem.
    """
    normalized_tokens = []
    for original_token in tokens:
        if stemming:
            normalized_token = PORTER_STEMMER.stem(original_token)
        else:
            normalized_token = original_token
        normalized_tokens.append(normalized_token)
    return normalized_tokens
