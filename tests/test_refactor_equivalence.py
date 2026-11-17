"""Numerical comparison against unmodified source from the pre-refactor commit."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
import torch

from tests.test_experiment_reproducibility import assert_same


BASELINE = "c52ca444263e88b26ccc173cc449224edb986cc6"
ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("refactor_reference_worker.py")


@pytest.fixture(scope="module")
def original_tree(tmp_path_factory):
    tree = tmp_path_factory.mktemp("pre-refactor")
    archive = subprocess.run(["git", "archive", BASELINE, "vitqat", "A3", "cropr"],
                             cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        # Extract files only, checking every path. Never alter the working tree.
        for member in source:
            destination = (tree / member.name).resolve()
            assert destination.is_relative_to(tree.resolve())
            if member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.extractfile(member).read())
    return tree


def run_worker(tree, version, name, output, *extra):
    completed = subprocess.run([sys.executable, str(WORKER), str(tree), version, name,
                                str(output), *extra], cwd=tree, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return torch.load(output / "result.pt", weights_only=True)


def old_state(state):
    # The new persistent metadata records bit width; it does not change arithmetic.
    return {key: value for key, value in state.items() if not key.endswith("._bit_width")}


# B1 is now Cropr + Linear QAT (LSQ), which has no pre-refactor counterpart in this tree.
@pytest.mark.parametrize("name", ["A1", "A2", "A4", "B2"])
@pytest.mark.parametrize("align_rng", [False, True], ids=["same-seed", "same-training-rng"])
def test_before_after_training(original_tree, tmp_path, name, align_rng):
    flags = ["--align-training-rng"] if align_rng else []
    before = run_worker(original_tree, "before", name, tmp_path / "before", *flags)
    after = run_worker(ROOT, "after", name, tmp_path / "after", *flags)
    initial_before = torch.load(tmp_path / "before/initial.pt", weights_only=True)
    initial_after = torch.load(tmp_path / "after/initial.pt", weights_only=True)
    assert_same(initial_before["model"], initial_after["model"])
    assert_same(before["records"], after["records"])
    assert_same(before["model"], old_state(after["model"]))
    assert_same(before["optimizer"], after["optimizer"])
    assert_same(before["rng"], after["rng"])


def test_before_after_a3_inference(original_tree, tmp_path):
    flags = ["--source-checkpoint", str(tmp_path / "a2.pt")]
    before = run_worker(original_tree, "before", "A3", tmp_path / "before", *flags)
    after = run_worker(ROOT, "after", "A3", tmp_path / "after", *flags)
    assert_same(before, after)
