from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest
from .data import prepare_voc
from .demo import create_demo
from .model import fit, predict
from .ranking import append_observation, rank_manifest


def _write(result, output: str | None = None) -> None:
    text = json.dumps(result, indent=2) + "\n"
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    print(text, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="use-first", description="Rank bananas by visible deterioration")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-voc", help="crop banana objects from the supported VOC dataset")
    prepare.add_argument("--source", required=True); prepare.add_argument("--output", required=True); prepare.add_argument("--seed", type=int, default=18)
    train = commands.add_parser("fit", help="fit and evaluate candidate stage classifiers")
    train.add_argument("--manifest", required=True); train.add_argument("--model", required=True); train.add_argument("--report"); train.add_argument("--seed", type=int, default=18)
    infer = commands.add_parser("predict", help="classify one banana crop")
    infer.add_argument("--model", required=True); infer.add_argument("--image", required=True); infer.add_argument("--output")
    rank = commands.add_parser("rank", help="rank item images from a CSV")
    rank.add_argument("--model", required=True); rank.add_argument("--items", required=True); rank.add_argument("--output"); rank.add_argument("--storage-weight", type=float, default=0.15)
    observe = commands.add_parser("observe", help="append a dated observation for one item")
    observe.add_argument("--model", required=True); observe.add_argument("--image", required=True); observe.add_argument("--item-id", required=True); observe.add_argument("--observed-at", required=True); observe.add_argument("--store", required=True)
    demo = commands.add_parser("make-demo", help="create clearly synthetic smoke-test images")
    demo.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="check prepared splits for duplicate image content")
    audit.add_argument("--manifest", required=True); audit.add_argument("--output"); audit.add_argument("--near-threshold", type=int, default=2)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-voc":
        _write(prepare_voc(args.source, args.output, args.seed))
    elif args.command == "fit":
        _write(fit(args.manifest, args.model, args.report, args.seed))
    elif args.command == "predict":
        _write(predict(args.model, args.image), args.output)
    elif args.command == "rank":
        if not 0 <= args.storage_weight <= 0.5:
            raise SystemExit("--storage-weight must be between 0 and 0.5")
        _write(rank_manifest(args.model, args.items, args.storage_weight), args.output)
    elif args.command == "observe":
        _write(append_observation(args.store, args.model, args.image, args.item_id, args.observed_at))
    elif args.command == "audit":
        _write(audit_manifest(args.manifest, args.output, args.near_threshold))
    else:
        _write({"synthetic_manifest": str(create_demo(args.output))})


if __name__ == "__main__":
    main()

