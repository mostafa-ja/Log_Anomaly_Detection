"""
Unit tests for template tokenization and semantic embeddings.
"""

import pytest
import numpy as np
from src.embeddings import TemplateTokenizer, TemplateEmbedder


def test_template_tokenizer():
    tokenizer = TemplateTokenizer()
    text = "Receiving block <*> src: 10.0.0.1 dest: 10.0.0.2 allocateBlock"
    tokens = tokenizer.tokenize(text)
    assert "receiving" in tokens
    assert "block" in tokens
    assert "allocate" in tokens
    assert "block" in tokens  # Split from allocateBlock


def test_template_embedder():
    embedder = TemplateEmbedder(n_components=5)
    templates = [
        "Receiving block <*> src: <*> dest: <*>",
        "BLOCK NameSystem allocateBlock: <*> <*>",
        "PacketResponder <*> for block <*> <*>",
        "Received block <*> of size <*> from <*>",
        "BLOCK NameSystem addStoredBlock blockMap updated",
        "Verification succeeded for <*>",
    ]
    matrix = embedder.fit_transform(templates)
    assert matrix.shape == (6, 5)

    # Test transform on unseen template
    new_template = ["Deleting block <*> file <*>"]
    new_vec = embedder.transform(new_template)
    assert new_vec.shape == (1, 5)
