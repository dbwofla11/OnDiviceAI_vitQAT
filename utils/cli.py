"""Only I/O and execution-environment arguments; no training recipe or inference switch."""
import argparse
import torch


def parser(description=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--experiment", help="Optional result label")
    p.add_argument("--data-root", default="data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument("--latency-repeats", type=int, default=100)
    return p
