#!/usr/bin/env python3
"""Regenerate the golden expectation file.

Run this ONLY when a model's weights or runtime intentionally change, and
record why in the commit message. A golden diff you did not intend is a bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mival.adapters import get_adapter
from mival.compiler import compile_recipe
from mival.modelcard import load_card
from mival.synthetic import GOLDEN_SOURCE, synthetic_ecg

N_RECORDS = 10


def probabilities(model_id: str, registry: Path) -> np.ndarray:
    card = load_card(registry / f"{model_id}.json")
    chain = compile_recipe(card.input_contract, GOLDEN_SOURCE)
    batch = np.stack(
        [chain.apply(synthetic_ecg(i)).data for i in range(N_RECORDS)]
    ).astype(np.float32)
    adapter = get_adapter(card.adapter)
    return adapter.forward(adapter.load(card), batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--registry", type=Path, default=Path("registry/models"))
    parser.add_argument("--out", type=Path, default=Path("tests/golden/expected.json"))
    args = parser.parse_args()

    probs = np.round(probabilities(args.model_id, args.registry), 6)
    body = json.loads(args.out.read_text()) if args.out.exists() else {}
    body[args.model_id] = {
        "n_records": N_RECORDS,
        "probabilities": [float(p) for p in probs],
        "sha256": hashlib.sha256(probs.tobytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body[args.model_id], indent=2))


if __name__ == "__main__":
    main()
