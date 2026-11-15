"""Write caller-defined result mappings. No model, task or framework assumptions."""
import csv
import json
from pathlib import Path


def save_results(metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document = json.dumps(metrics, indent=2)
    (output / "metrics.json").write_text(document, encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(document, flush=True)
    return metrics
