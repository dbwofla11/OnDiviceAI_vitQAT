"""The common writer works without ML packages or a classification report schema."""
import csv
import json
from pathlib import Path
import subprocess
import sys


def test_writer_without_site_packages(tmp_path):
    script = '''
import sys
from utils.results import save_results
metrics = {"run": "custom-model", "quality": 0.875, "artifact_bytes": 123}
before = dict(metrics)
assert save_results(metrics, sys.argv[1]) == before
assert metrics == before
assert not {"torch", "torchvision", "timm"}.intersection(sys.modules)
'''
    output = tmp_path / "report"
    subprocess.run([sys.executable, "-S", "-c", script, str(output)],
                   cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    assert json.loads((output / "metrics.json").read_text()) == {
        "run": "custom-model", "quality": 0.875, "artifact_bytes": 123,
    }
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"run": "custom-model", "quality": "0.875", "artifact_bytes": "123"},
        ]
    assert {p.name for p in output.iterdir()} == {"metrics.json", "metrics.csv"}
