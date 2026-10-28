"""
Semantic Embedding Module
=========================
Extracts semantic dense representations from log event templates using
tokenization, TF-IDF / Word embeddings, and PCA dimensionality reduction.
"""

import os
import re
import json
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


class TemplateTokenizer:
    """Tokenizes and normalizes log templates for semantic representation."""

    DEFAULT_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with"
    }

    def __init__(self, custom_stopwords: Optional[set] = None):
        self.stopwords = custom_stopwords if custom_stopwords is not None else self.DEFAULT_STOPWORDS
        self.special_char_pattern = re.compile(r"[^\w\s]")
        self.camel_case_pattern = re.compile(r"([a-z])([A-Z])")

    def tokenize(self, template: str) -> List[str]:
        """Normalize template string: remove special chars, split camelCase, lowercase."""
        text = template.replace("<*>", " ")
        text = self.camel_case_pattern.sub(r"\1 \2", text)
        text = self.special_char_pattern.sub(" ", text)
        tokens = [
            t.lower()
            for t in text.split()
            if len(t) > 1 and not t.isdigit() and t.lower() not in self.stopwords
        ]
        return tokens

    def clean_text(self, template: str) -> str:
        """Return cleaned normalized sentence string."""
        return " ".join(self.tokenize(template))


class TemplateEmbedder:
    """Computes dense semantic embeddings for templates with PCA dimensionality reduction."""

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.tokenizer = TemplateTokenizer()
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit_transform(self, templates: List[str]) -> np.ndarray:
        """Fit TF-IDF and PCA on template list, then return dense vectors."""
        cleaned_docs = [self.tokenizer.clean_text(t) for t in templates]
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs).toarray()

        actual_components = min(self.n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
        if actual_components < self.n_components:
            self.pca = PCA(n_components=actual_components, random_state=42)

        reduced_matrix = self.pca.fit_transform(tfidf_matrix)

        # Pad with zeros if actual components < requested n_components
        if reduced_matrix.shape[1] < self.n_components:
            padding = np.zeros((reduced_matrix.shape[0], self.n_components - reduced_matrix.shape[1]))
            reduced_matrix = np.hstack([reduced_matrix, padding])

        self.is_fitted = True
        return reduced_matrix

    def transform(self, templates: List[str]) -> np.ndarray:
        """Transform new templates using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("TemplateEmbedder must be fitted before calling transform.")
        cleaned_docs = [self.tokenizer.clean_text(t) for t in templates]
        tfidf_matrix = self.vectorizer.transform(cleaned_docs).toarray()
        reduced_matrix = self.pca.transform(tfidf_matrix)

        if reduced_matrix.shape[1] < self.n_components:
            padding = np.zeros((reduced_matrix.shape[0], self.n_components - reduced_matrix.shape[1]))
            reduced_matrix = np.hstack([reduced_matrix, padding])

        return reduced_matrix

    def save(self, filepath: str, embeddings: np.ndarray):
        """Save embeddings to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(embeddings.tolist(), f)

    @staticmethod
    def load(filepath: str) -> np.ndarray:
        """Load embeddings from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return np.array(data, dtype=np.float32)
