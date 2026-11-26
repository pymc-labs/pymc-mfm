"""Turn a local dated CSV and budget configuration into a budget allocation."""

import argparse
import json
from pathlib import Path

from allocate import allocate, load_model

from pymc_mfm.budget.history import read_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="JSON budget and optional limits")
    p.add_argument(
        "--channel", action="append", required=True, help="spend column, in output order"
    )
    p.add_argument("--outcome", required=True, help="aggregate outcome column")
    p.add_argument(
        "--control", action="append", default=[], help="observed historical control column"
    )
    p.add_argument("--date", default="date", help="ISO date column (default: date)")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        p.error("config must be a JSON object")
    if set(config) & {"channels", "spend", "outcome", "controls"}:
        p.error("config may not override CSV history or channel order")
    history = read_history(
        args.history,
        channels=args.channel,
        outcome=args.outcome,
        controls=args.control,
        date_column=args.date,
    )
    print(json.dumps(allocate(load_model(args.model), {**config, **history}), indent=2))


if __name__ == "__main__":
    main()
