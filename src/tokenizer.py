"""Shared query and document tokenization."""
import re


def tokenize(text: str) -> list[str]:
    """Lowercase and split on punctuation and whitespace; retain letters/digits."""
    return re.findall(r"[^\W_]+", text.lower())
