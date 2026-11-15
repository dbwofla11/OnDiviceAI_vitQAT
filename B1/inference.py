"""B1 checkpoint inference. Does not import the training entry point."""
from utils.data import make_food101_data as make_data, loader
from utils.reproducibility import configure_backend
from utils.evaluation import run
from utils.classification import evaluate as evaluate_logits
from .config import parser as config_parser, validate, CHECKPOINT_DEFAULTS
from .model import build_model, restore_model
from .metrics import measure_model, report_results


def parser():
    return config_parser(inference=True)


def evaluate(model, batches, device):
    return evaluate_logits(model, batches, device)


def main(args):
    configure_backend(deterministic=True, allow_tf32=False)
    return run(args, build_model=build_model, evaluate_fn=evaluate,
               measure_model=measure_model, report_results=report_results,
               data_factory=make_data, loader_factory=loader, validate_config=validate,
               config_defaults=CHECKPOINT_DEFAULTS, restore_model=restore_model)


if __name__ == "__main__":
    main(parser().parse_args())
