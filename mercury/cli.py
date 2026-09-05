"""Command-line interface for the Mercury flywheel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mercury.demo import frontier_auth_fix, frontier_pytest_fix, lesser_auth_fail
from mercury.harness import MercuryHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mercury",
        description="Capture frontier agent operations and embed them into lesser-model sessions.",
    )
    parser.add_argument("--store", default=".mercury", help="Knowledge store directory (default: .mercury)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create an empty knowledge store")

    capture = sub.add_parser("capture", help="Ingest a frontier (or other) agent trace JSON file")
    capture.add_argument("trace", help="Path to a Mercury, OpenAI, or Cursor-like trace JSON file")
    capture.add_argument("--teacher", action="store_true", help="Treat this trace as a teacher even if the model is not frontier")

    sub.add_parser("distill", help="Re-distill every stored trace into operational cards")

    pack = sub.add_parser("pack", help="Print a Frontier Operating Pack for a lesser-model task")
    pack.add_argument("--task", required=True, help="The student model's upcoming task")
    pack.add_argument("--model", default="gpt-4o-mini", help="Student model name")
    pack.add_argument("--error", dest="error", default=None, help="Optional current error signature")
    pack.add_argument("--format", choices=("markdown", "cursor-rule", "json"), default="markdown")

    contrast = sub.add_parser("contrast", help="Distill divergence between a failed lesser run and a successful frontier run")
    contrast.add_argument("student", help="Failed lesser-model trace JSON")
    contrast.add_argument("teacher", help="Successful frontier trace JSON")

    sub.add_parser("status", help="Show store statistics")
    grade = sub.add_parser(
        "grade",
        help="Deterministically grade how a trace operated (policy floor + competence ceiling)",
    )
    grade.add_argument("trace", help="Path to a Mercury, OpenAI, or Cursor-like trace JSON file")
    grade.add_argument(
        "--compare",
        dest="compare",
        default=None,
        help="Optional second trace: print grade delta (after − before). First trace is before.",
    )
    grade.add_argument("--json", action="store_true", help="Print the grade as JSON")
    demo = sub.add_parser("demo", help="Run the built-in frontier → lesser flywheel on fixture traces")
    demo.add_argument("--json", action="store_true", help="Print pack as JSON")

    args = parser.parse_args(argv)
    harness = MercuryHarness.init(args.store)

    if args.command == "init":
        print(f"Initialized Mercury store at {Path(args.store).resolve()}")
        return 0

    if args.command == "capture":
        trace = harness.capture(args.trace, force_teacher=args.teacher)
        stats = harness.stats()
        print(f"Captured {trace.id} ({trace.model}, {trace.tier.value}, {trace.outcome.status.value})")
        print(f"Store now has {stats['traces']} traces and {stats['cards']} cards")
        return 0

    if args.command == "distill":
        count = harness.distill_all()
        print(f"Distilled {count} operational cards")
        return 0

    if args.command == "pack":
        pack_obj = harness.pack(args.task, model=args.model, error_signature=args.error)
        _print_pack(pack_obj, args.format)
        return 0

    if args.command == "contrast":
        cards = harness.contrast(args.student, args.teacher)
        print(f"Wrote {len(cards)} contrastive/teacher cards")
        return 0

    if args.command == "grade":
        from mercury.grade import grade_delta, grade_trace
        from mercury.traceio import load_trace

        if args.compare:
            delta = grade_delta(load_trace(args.trace), load_trace(args.compare))
            if args.json:
                json.dump(delta.as_dict(), sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print(delta.summary())
                print()
                print("Before:")
                print(delta.before.summary())
                print()
                print("After:")
                print(delta.after.summary())
            return 0

        report = grade_trace(load_trace(args.trace))
        if args.json:
            json.dump(report.as_dict(), sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(report.summary())
        return 0

    if args.command == "status":
        stats = harness.stats()
        print(json.dumps(stats, indent=2, default=str))
        return 0

    if args.command == "demo":
        return _run_demo(harness, as_json=args.json)

    return 1


def _print_pack(pack, fmt: str) -> None:
    if fmt == "markdown":
        sys.stdout.write(pack.render())
        return
    if fmt == "cursor-rule":
        sys.stdout.write(pack.as_cursor_rule())
        return
    payload = {
        "task": pack.task,
        "model": pack.model,
        "tier": pack.tier.value,
        "cards": [card.model_dump() for card in pack.cards],
        "scores": pack.scores,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _run_demo(harness: MercuryHarness, *, as_json: bool) -> int:
    teacher = frontier_auth_fix()
    extra = frontier_pytest_fix()
    student = lesser_auth_fail()
    harness.capture(teacher)
    harness.capture(extra)
    harness.contrast(student, teacher)
    pack = harness.pack(
        "Users bounce back to /login after authenticating. Fix the redirect bug.",
        model="gpt-4o-mini",
        languages=["typescript"],
    )
    stats = harness.stats()
    if as_json:
        json.dump(
            {"stats": stats, "pack": [card.title for card in pack.cards], "markdown": pack.render()},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print("Mercury demo: frontier traces captured, knowledge embedded, lesser-model pack follows.\n")
        print(f"Store: {stats['traces']} traces, {stats['cards']} cards, kinds={stats['by_kind']}")
        print()
        sys.stdout.write(pack.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
