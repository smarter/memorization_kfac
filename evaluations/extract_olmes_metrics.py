#!/usr/bin/env python
"""Extract primary scores from a benchmark metrics.json to a simpler format for DVC.

Accepts two schemas:
  OLMES (oe-eval):  {"tasks": [{"alias": "gsm8k::olmes", "metrics": {"primary_score": 0.674}}]}
  olmo-eval:        {"summary": {"gsm8k": {"metric": "accuracy:exact_match", "score": 0.674}}, ...}

Writes {alias: score}. For the olmo-eval schema the alias is taken from --alias
(default: the olmo-eval task name), so a task evaluated by olmo-eval with the
same formulation as an OLMES task can keep the OLMES metric key.

Usage:
    python extract_olmes_metrics.py input.json output.json [--alias gsm8k::olmes]
"""

import argparse
import json


def extract_primary_scores(input_path: str, output_path: str, alias: str | None = None) -> None:
    with open(input_path) as f:
        data = json.load(f)

    if "tasks" in data and data["tasks"] and "alias" in data["tasks"][0]:  # OLMES
        scores = {task["alias"]: task["metrics"]["primary_score"] for task in data["tasks"]}
    elif "summary" in data:  # olmo-eval
        summary = data["summary"]
        if alias is not None and len(summary) == 1:
            scores = {alias: next(iter(summary.values()))["score"]}
        else:
            scores = {name: entry["score"] for name, entry in summary.items()}
    else:
        raise ValueError(f"Unrecognised metrics schema in {input_path}: keys {list(data)}")

    with open(output_path, "w") as f:
        json.dump(scores, f)


def main():
    parser = argparse.ArgumentParser(description="Extract primary scores from a benchmark metrics.json")
    parser.add_argument("input", help="Path to metrics.json (OLMES or olmo-eval)")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument("--alias", default=None, help="Metric key to use for a single olmo-eval task")
    args = parser.parse_args()

    extract_primary_scores(args.input, args.output, args.alias)


if __name__ == "__main__":
    main()
