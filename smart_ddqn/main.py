from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Add project roots before importing Habitat-specific modules.
    for value in reversed(cfg["run"].get("python_paths", [])):
        path = str(Path(value).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    from trainer import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
