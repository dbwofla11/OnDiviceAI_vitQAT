"""A3 paired dense/pruned checkpoint inference."""
import json
from .eval_pruning import parser, run


def main(args):
    return run(args)


if __name__ == "__main__":
    print(json.dumps(main(parser().parse_args()), indent=2))
