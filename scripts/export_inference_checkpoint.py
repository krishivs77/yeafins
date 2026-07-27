"""Export a deterministic, inference-only Yeafins policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from yeafins.checkpoints import export_inference_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    export_inference_checkpoint(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
