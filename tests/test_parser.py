"""
Unit tests for Drain log parser.
"""

import pytest
import pandas as pd
from src.parser import DrainParser, LogCluster


def test_drain_parser_initialization():
    parser = DrainParser(log_format="<Date> <Time> <Content>", depth=4, similarity_threshold=0.5)
    assert parser.depth == 2
    assert parser.st == 0.5
    assert len(parser.log_clusters) == 0


def test_drain_parse_single_line():
    parser = DrainParser(log_format="<Content>", depth=4, similarity_threshold=0.5)
    t1, id1, params1 = parser.parse_line("Receiving block blk_-12345 src: 10.0.0.1 dest: 10.0.0.2", 0)
    assert "<*>" in t1
    assert len(id1) == 8

    # Second line with same template should match
    t2, id2, params2 = parser.parse_line("Receiving block blk_-99999 src: 192.168.1.1 dest: 192.168.1.2", 1)
    assert id1 == id2
    assert len(parser.log_clusters) == 1


def test_drain_dataframe_parsing():
    parser = DrainParser(log_format="<Content>", depth=4, similarity_threshold=0.5)
    df = pd.DataFrame({
        "Content": [
            "User admin logged in from 10.0.0.1",
            "User guest logged in from 10.0.0.2",
            "Disk failure on drive /dev/sda1",
        ]
    })
    df_parsed = parser.parse_dataframe(df, content_col="Content")
    assert "EventTemplate" in df_parsed.columns
    assert "EventId" in df_parsed.columns
    assert len(df_parsed) == 3

    df_templates = parser.get_template_df()
    assert len(df_templates) >= 2
