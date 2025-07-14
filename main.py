from Classifier import *
from Clustering import *
from DataLoader import *

import argparse
import tomli


def load_config(path: str) -> dict:
    with open(f"./configs/{path}", "rb") as f:
        return tomli.load(f)

def parse_args_with_config():
    parser = argparse.ArgumentParser(description="Clustering with TOML defaults", add_help=False)
    parser.add_argument('--config', type=str, default="default.toml")
    args, remaining_args = parser.parse_known_args()

    config = load_config(args.config)

    full_parser = argparse.ArgumentParser(description="Full parser with TOML default")
    full_parser = argparse.ArgumentParser(description="Full parser with TOML default")
    full_parser.add_argument('--method', type=str, default=config["cluster"]["method"])
    full_parser.add_argument('--column', type=str, default=config["cluster"]["column"])
    full_parser.add_argument('--config', type=str, default=args.config)

    return full_parser.parse_args(remaining_args)

if __name__ == "__main__":
    args = parse_args_with_config()
    print(f"Using config: {args.config}")
    print(f"Method: {args.method}")
    print(f"Column: {args.column}")