import pytest
import torch

from tests.support import parser
from tests.support import build_model
from tests.support import main


@pytest.mark.parametrize("mode,weight_bits,activation_bits", [("fp32", 2, 2), ("qat", 2, 2), ("qat", 4, 8)])
def test_precision_options(mode, weight_bits, activation_bits):
    args = parser().parse_args(["--mode", mode, "--output", "runs/example",
                               "--weight-bits", str(weight_bits), "--activation-bits", str(activation_bits)])
    assert (args.mode, args.weight_bits, args.activation_bits) == (mode, weight_bits, activation_bits)


def test_model_is_selected_by_option():
    torch.set_num_threads(2)
    args = parser().parse_args(["--mode", "fp32", "--model", "vit_small_patch16_224", "--output", "runs/example"])
    model = build_model(args)
    assert model.embed_dim == 384  # Tiny uses 192.
    assert model.head.out_features == 101


def test_invalid_freeze_is_rejected_before_creating_run(tmp_path):
    out = tmp_path / "invalid"
    args = parser().parse_args(["--mode", "qat", "--output", str(out), "--freeze-observers-epoch", "0"])
    with pytest.raises(ValueError, match="Observers"):
        main(args)
    assert not out.exists()
