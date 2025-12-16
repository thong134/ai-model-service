from __future__ import annotations

import re
import unicodedata
from typing import List

import emoji
from underthesea import word_tokenize


WHITESPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(
    r"([a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ])\1{2,}"
)
PUNCT_RE = re.compile(r"[^\w\s]")
NUMBER_RE = re.compile(r"\d+")


def normalize_repeated_characters(text: str) -> str:
    return REPEAT_CHAR_RE.sub(r"\1\1", text)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def basic_clean(text: str) -> str:
    text = text.lower()
    text = emoji.replace_emoji(text, replace=" ")
    text = PUNCT_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = normalize_repeated_characters(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    cleaned = basic_clean(text)
    if not cleaned:
        return []
    tokenized = word_tokenize(cleaned, format="text")
    tokens = [token for token in tokenized.split(" ") if token]
    return tokens


def preprocess_for_vectorizer(text: str, strip_diacritics: bool = False) -> str:
    tokens = tokenize(text)
    joined = " ".join(tokens)
    if strip_diacritics:
        joined = strip_accents(joined)
    return joined


def preprocess_for_tfidf(text: str) -> str:
    """Helper exposed for pickled TF-IDF pipelines."""
    return preprocess_for_vectorizer(text, strip_diacritics=False)
