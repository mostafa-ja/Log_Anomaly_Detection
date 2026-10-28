"""
Drain Log Parser Module
=======================
Implementation of the Drain algorithm for automated online log parsing.
References:
- He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017).
  Drain: An online log parser with fixed depth tree.
  IEEE International Conference on Web Services (ICWS).
"""

import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class LogCluster:
    """Represents a cluster of log messages mapping to a single template."""

    def __init__(self, log_template: List[str] = None, log_ids: List[int] = None):
        self.log_template = log_template if log_template is not None else []
        self.log_ids = log_ids if log_ids is not None else []


class DrainNode:
    """A node in the Drain prefix tree."""

    def __init__(self, depth: int = 0, token: Optional[str] = None):
        self.depth = depth
        self.token = token
        self.children: Dict[any, any] = {}


class DrainParser:
    """
    Online Tree-based Log Parser using Drain.
    Parses unstructured raw log messages into structured event templates and parameters.
    """

    def __init__(
        self,
        log_format: str,
        depth: int = 4,
        similarity_threshold: float = 0.5,
        max_children: int = 100,
        regex_patterns: Optional[List[str]] = None,
        keep_parameters: bool = True,
    ):
        """
        Parameters
        ----------
        log_format : str
            Format string, e.g., '<Date> <Time> <Pid> <Level> <Component>: <Content>'
            or '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
        depth : int
            Depth of prefix tree (default: 4)
        similarity_threshold : float
            Token similarity threshold (default: 0.5)
        max_children : int
            Maximum branches per internal node before grouping into wildcards (default: 100)
        regex_patterns : list of str
            Regular expressions to mask dynamic parameters (IPs, block IDs, numbers)
        keep_parameters : bool
            Whether to extract and record variable parameters
        """
        self.log_format = log_format
        self.depth = depth - 2
        self.st = similarity_threshold
        self.max_children = max_children
        self.keep_para = keep_parameters
        self.regex = regex_patterns if regex_patterns is not None else [
            r"blk_-?\d+",                      # HDFS block ids
            r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP addresses and ports
            r"(0x)?[0-9a-fA-F]{4,}",            # Hex addresses / hashes
            r"\b\d+\b",                        # Standalone numbers
        ]
        self.root_node = DrainNode()
        self.log_clusters: List[LogCluster] = []

    def _has_numbers(self, s: str) -> bool:
        return any(char.isdigit() for char in s)

    def _fast_match(self, cluster_list: List[LogCluster], tokens: List[str]) -> Optional[LogCluster]:
        """Find the most similar cluster in the leaf node candidate list."""
        max_sim = -1.0
        max_num_same = 0
        best_cluster = None

        for cluster in cluster_list:
            cur_sim, num_same = self._calculate_similarity(cluster.log_template, tokens)
            if cur_sim > max_sim or (cur_sim == max_sim and num_same > max_num_same):
                max_sim = cur_sim
                max_num_same = num_same
                best_cluster = cluster

        if max_sim >= self.st:
            return best_cluster
        return None

    def _calculate_similarity(self, template: List[str], tokens: List[str]) -> Tuple[float, int]:
        """Calculate token similarity between template and log tokens."""
        if len(template) != len(tokens) or len(tokens) == 0:
            return 0.0, 0
        same_tokens = sum(1 for t, w in zip(template, tokens) if t == w)
        return same_tokens / len(tokens), same_tokens

    def _tree_search(self, tokens: List[str]) -> Optional[LogCluster]:
        """Traverse prefix tree to retrieve candidate clusters."""
        seq_len = len(tokens)
        if seq_len not in self.root_node.children:
            return None

        current_node = self.root_node.children[seq_len]
        current_depth = 1

        for token in tokens:
            if current_depth >= self.depth or current_depth > seq_len:
                break
            if token in current_node.children:
                current_node = current_node.children[token]
            elif "<*>" in current_node.children:
                current_node = current_node.children["<*>"]
            else:
                return None
            current_depth += 1

        return self._fast_match(current_node.children, tokens)

    def _add_cluster(self, cluster: LogCluster):
        """Insert a newly discovered log cluster into the fixed-depth tree."""
        tokens = cluster.log_template
        seq_len = len(tokens)

        if seq_len not in self.root_node.children:
            layer1 = DrainNode(depth=1, token=str(seq_len))
            self.root_node.children[seq_len] = layer1
        else:
            layer1 = self.root_node.children[seq_len]

        current_node = layer1
        current_depth = 1

        for token in tokens:
            if current_depth >= self.depth or current_depth > seq_len:
                if len(current_node.children) == 0:
                    current_node.children = [cluster]
                else:
                    current_node.children.append(cluster)
                break

            if token not in current_node.children:
                if not self._has_numbers(token):
                    if "<*>" in current_node.children:
                        if len(current_node.children) < self.max_children:
                            new_node = DrainNode(depth=current_depth + 1, token=token)
                            current_node.children[token] = new_node
                            current_node = new_node
                        else:
                            current_node = current_node.children["<*>"]
                    else:
                        if len(current_node.children) + 1 < self.max_children:
                            new_node = DrainNode(depth=current_depth + 1, token=token)
                            current_node.children[token] = new_node
                            current_node = new_node
                        elif len(current_node.children) + 1 == self.max_children:
                            new_node = DrainNode(depth=current_depth + 1, token="<*>")
                            current_node.children["<*>"] = new_node
                            current_node = new_node
                        else:
                            current_node = current_node.children["<*>"]
                else:
                    if "<*>" not in current_node.children:
                        new_node = DrainNode(depth=current_depth + 1, token="<*>")
                        current_node.children["<*>"] = new_node
                        current_node = new_node
                    else:
                        current_node = current_node.children["<*>"]
            else:
                current_node = current_node.children[token]

            current_depth += 1

    def _mask_tokens(self, content: str) -> str:
        """Apply regex mask rules to substitute parameters with wildcard."""
        masked = content
        for pattern in self.regex:
            masked = re.sub(pattern, "<*>", masked)
        return masked

    def _extract_parameters(self, template: List[str], tokens: List[str]) -> List[str]:
        """Extract matched parameters where template has <*>."""
        params = []
        for t, w in zip(template, tokens):
            if t == "<*>":
                params.append(w)
        return params

    def parse_line(self, content: str, line_id: int) -> Tuple[str, str, List[str]]:
        """
        Parse a single log message content string.
        Returns: (event_template_str, event_id_hash, parameter_list)
        """
        masked_content = self._mask_tokens(content.strip())
        tokens = [t for t in masked_content.split() if t]
        if not tokens:
            tokens = ["<EMPTY>"]

        matched_cluster = self._tree_search(tokens)
        if matched_cluster is None:
            new_cluster = LogCluster(log_template=list(tokens), log_ids=[line_id])
            self.log_clusters.append(new_cluster)
            self._add_cluster(new_cluster)
            template_tokens = tokens
        else:
            # Update template with wildcards where tokens differ
            for i, (t_tok, l_tok) in enumerate(zip(matched_cluster.log_template, tokens)):
                if t_tok != l_tok:
                    matched_cluster.log_template[i] = "<*>"
            matched_cluster.log_ids.append(line_id)
            template_tokens = matched_cluster.log_template

        template_str = " ".join(template_tokens)
        event_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[:8]
        params = self._extract_parameters(template_tokens, tokens) if self.keep_para else []
        return template_str, event_id, params

    def parse_dataframe(self, df: pd.DataFrame, content_col: str = "Content") -> pd.DataFrame:
        """Parse an entire DataFrame in-memory."""
        templates = []
        event_ids = []
        parameters = []

        for idx, row in df.iterrows():
            content = str(row[content_col])
            t_str, e_id, params = self.parse_line(content, idx)
            templates.append(t_str)
            event_ids.append(e_id)
            parameters.append(params)

        df_out = df.copy()
        df_out["EventTemplate"] = templates
        df_out["EventId"] = event_ids
        df_out["ParameterList"] = parameters
        return df_out

    def get_template_df(self) -> pd.DataFrame:
        """Return unique discovered event templates with occurrence counts."""
        records = []
        for cluster in self.log_clusters:
            t_str = " ".join(cluster.log_template)
            e_id = hashlib.md5(t_str.encode("utf-8")).hexdigest()[:8]
            records.append({
                "EventId": e_id,
                "EventTemplate": t_str,
                "Occurrences": len(cluster.log_ids)
            })
        df_templates = pd.DataFrame(records)
        if not df_templates.empty:
            df_templates = df_templates.sort_values(by="Occurrences", ascending=False).reset_index(drop=True)
        return df_templates
