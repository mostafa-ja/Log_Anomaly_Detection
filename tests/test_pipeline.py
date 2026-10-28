"""
Integration test running a quick end-to-end training and evaluation cycle.
"""

import pytest
import subprocess
import sys


def test_quick_train_eval_deeplog():
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--model",
        "deeplog",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--window-size",
        "5",
        "--num-sessions",
        "100",
        "--num-classes",
        "15",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error output: {result.stderr}"
    assert "Benchmark Results" in result.stdout
