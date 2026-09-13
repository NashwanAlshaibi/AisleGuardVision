#!/usr/bin/env python3
"""Run behavior scenarios through the real pipeline, with no neural networks.

This is the most important development tool in the repository. Behavior logic
and model quality are independent problems: coupling them means every tuning
change needs a GPU, weights and footage, and every regression is ambiguous
between "the detector got worse" and "the logic got worse".

Scenarios emit scripted detections that are driven through the *production*
tracker, associator and behavior engine. Only the models are absent.

    python scripts/simulate_behavior.py
    python scripts/simulate_behavior.py --scenario POSSIBLE_CONCEALMENT --verbose
    python scripts/simulate_behavior.py --timeline --scenario ITEM_TO_BASKET
    python scripts/simulate_behavior.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aisleguardvision.core.config import AppConfig, load_config  # noqa: E402
from aisleguardvision.core.logging import configure_logging  # noqa: E402
from aisleguardvision.simulation.harness import SimulationResult, SimulationRunner  # noqa: E402
from aisleguardvision.simulation.scenarios import SCENARIOS, all_scenarios, get_scenario  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def print_result(result: SimulationResult, verbose: bool, timeline: bool) -> None:
    scenario = result.scenario
    status = "PASS" if result.matches_expectation else "FAIL"

    print(RULE)
    print(f"  {scenario.name}   [{status}]")
    print(RULE)
    print(f"  {scenario.description}")
    print(f"  Why this scenario exists: {scenario.rationale}")
    print()
    print(f"  expected : {scenario.expected}")
    print(f"  actual   : {result.outcome}")
    print(f"  peak risk: {result.peak_risk:.1f}  ({result.peak_threat.value})")
    print(f"  peak state: {result.peak_state.value}")
    print(f"  frames   : {len(result.outcomes)}  duration: "
          f"{result.outcomes[-1].timestamp if result.outcomes else 0:.2f}s")
    if not scenario.item_detection_available:
        print("  note     : running WITHOUT a merchandise detector (zone-only mode)")

    print()
    print("  State transitions:")
    if result.transitions:
        for timestamp, person_id, previous, current in result.transitions:
            print(f"    {timestamp:6.2f}s  track {person_id}:  {previous.value} -> {current.value}")
    else:
        print("    (none - remained IDLE)")

    positive, negative = result.evidence_summary()
    print()
    print("  Evidence for concealment (at peak risk):")
    if positive:
        for text in positive:
            print(f"    + {text}")
    else:
        print("    (none)")
    print()
    print("  Evidence against:")
    if negative:
        for text in negative:
            print(f"    - {text}")
    else:
        print("    (none)")

    if timeline:
        print()
        print("  Risk timeline:")
        previous_score = None
        for outcome in result.outcomes:
            for observation in outcome.observations:
                score = observation.risk.risk_score
                if previous_score is not None and abs(score - previous_score) < 0.05:
                    continue
                previous_score = score
                bar = "#" * int(score / 2.5)
                print(
                    f"    {outcome.timestamp:6.2f}s  {score:6.1f}  "
                    f"{observation.state.value:<26} {bar}"
                )

    if verbose and result.outcomes:
        print()
        print("  Full risk breakdown at peak:")
        best = None
        for outcome in result.outcomes:
            for observation in outcome.observations:
                if best is None or observation.risk.risk_score > best.risk.risk_score:
                    best = observation
        if best is not None:
            for line in best.risk.explain().splitlines():
                print(f"    {line}")

    print()
    print(f"  RESULT: {result.outcome}")
    if result.alerted:
        print("          An incident would be created for HUMAN REVIEW.")
        print("          This is NOT a determination that a theft occurred.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scenario", "-s", action="append", choices=sorted(SCENARIOS), help="Scenario (repeatable)"
    )
    parser.add_argument("--config-dir", default=None, help="Configuration directory to load")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show the full risk breakdown")
    parser.add_argument("--timeline", "-t", action="store_true", help="Show the risk timeline")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--log-level", default="ERROR", help="Log level for the engine itself")
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, fmt="text", force=True)

    config = load_config(args.config_dir) if args.config_dir else AppConfig()
    scenarios = [get_scenario(name) for name in args.scenario] if args.scenario else all_scenarios()

    runner = SimulationRunner(config)
    results = [runner.run(scenario) for scenario in scenarios]

    if args.json:
        payload = [
            {
                "scenario": r.scenario.name,
                "expected": r.scenario.expected,
                "actual": r.outcome,
                "passed": r.matches_expectation,
                "peak_risk": round(r.peak_risk, 2),
                "peak_threat": r.peak_threat.value,
                "peak_state": r.peak_state.value,
                "transitions": [
                    {
                        "t": round(t, 3),
                        "person_id": pid,
                        "from": a.value,
                        "to": b.value,
                    }
                    for t, pid, a, b in r.transitions
                ],
                "positive_evidence": r.evidence_summary()[0],
                "negative_evidence": r.evidence_summary()[1],
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
        return 0 if all(r.matches_expectation for r in results) else 1

    print()
    print("AisleGuard Vision - behavior simulation")
    print("Scripted detections through the real tracking and behavior stack.")
    print(f"Alert threshold: {config.behavior.alert_threshold:.0f}")
    print()

    for result in results:
        print_result(result, args.verbose, args.timeline)

    passed = sum(1 for r in results if r.matches_expectation)
    print(THIN)
    print(f"  {passed}/{len(results)} scenarios produced their expected outcome")
    print(THIN)
    for result in results:
        mark = "PASS" if result.matches_expectation else "FAIL"
        print(
            f"  {mark}  {result.scenario.name:<36} "
            f"expected={result.scenario.expected:<9} actual={result.outcome:<9} "
            f"peak_risk={result.peak_risk:6.1f}"
        )
    print()
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
