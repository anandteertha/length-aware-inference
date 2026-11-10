"""Demonstrate the original frozen CPU policy with reusable local sessions."""

import argparse
import json
from pathlib import Path

from etdl.config import load_config
from etdl.io import sha256, write_json
from etdl.router import RoutedClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--text", nargs="+", default=["Great service!", "This was a terrible movie."]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    policy = args.policy or config.artifact_dir / "confirmation/frozen-policy.json"
    classifier = RoutedClassifier(config, policy)
    predictions = [{"text": text, **classifier.predict(text)} for text in args.text]
    report = {
        "purpose": "Functional demonstration; not an accuracy or latency benchmark.",
        "policy_sha256": sha256(policy),
        "threads": classifier.policy["threads"],
        "retained_variants": classifier.policy["variants"],
        "ready_ms": classifier.ready_ms,
        "predictions": predictions,
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
