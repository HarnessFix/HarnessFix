#!/usr/bin/env python3
"""Regression-aware validation gate with target metrics, regressions, and cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from failure_analysis.validation_metrics import (  # noqa: E402
    compute_cost_ratio,
    compute_run_metrics,
    evaluate_target_metrics,
    load_eval_json,
)


def _resolved_ids(eval_json: Path, subset: set[str]) -> set[str]:
    data = json.loads(eval_json.read_text())
    return set(data.get("resolved_ids", [])) & subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate enhanced agent against a baseline gate")
    parser.add_argument("--mode", choices=["swe", "gaia", "appworld", "terminal_bench"], default="swe")
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--current-eval", type=Path, required=True)
    parser.add_argument("--baseline-traces", type=Path, default=None)
    parser.add_argument("--current-traces", type=Path, default=None)
    parser.add_argument("--plan-spec", type=Path, default=None)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--max-regression", type=int, default=2)
    parser.add_argument("--min-improvement", type=int, default=1,
                        help="Fallback resolved-count threshold when target metrics are unavailable")
    parser.add_argument("--min-target-improvement", type=float, default=0.0)
    parser.add_argument("--max-cost-ratio", type=float, default=1.25)
    args = parser.parse_args()

    subset = set(args.val_ids.read_text().split())
    baseline_path, _ = load_eval_json(args.baseline_eval)
    current_path, _ = load_eval_json(args.current_eval)

    baseline_resolved = _resolved_ids(baseline_path, subset)
    current_resolved = _resolved_ids(current_path, subset)
    regressed_ids = sorted(baseline_resolved - current_resolved)
    improved_ids = sorted(current_resolved - baseline_resolved)
    regression_count = len(regressed_ids)
    improvement_count = len(improved_ids)
    net_change = len(current_resolved) - len(baseline_resolved)

    plan_spec = {"fixes": []}
    if args.plan_spec and args.plan_spec.exists():
        plan_spec = json.loads(args.plan_spec.read_text())

    baseline_metrics = None
    current_metrics = None
    target_results = {"metrics": [], "aggregate_delta": 0.0, "improved_metric_count": 0}
    cost_ratio = 1.0
    resolved_pass = net_change >= args.min_improvement
    target_pass = True

    if args.baseline_traces and args.current_traces:
        baseline_metrics = compute_run_metrics(args.mode, args.baseline_traces, baseline_path)
        current_metrics = compute_run_metrics(args.mode, args.current_traces, current_path)
        target_results = evaluate_target_metrics(plan_spec, baseline_metrics, current_metrics)
        cost_ratio = compute_cost_ratio(baseline_metrics, current_metrics)
        target_pass = (
            target_results["improved_metric_count"] > 0
            and target_results["aggregate_delta"] >= args.min_target_improvement
        )

    regression_pass = regression_count <= args.max_regression
    passed = resolved_pass and target_pass and regression_pass

    failure_reasons = []
    if not resolved_pass:
        failure_reasons.append("insufficient_resolved_improvement")
    if not target_pass:
        failure_reasons.append("insufficient_target_improvement")
    if not regression_pass:
        failure_reasons.append("excessive_regressions")

    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 72}")
    print(f"Validation Gate: {status}")
    print(f"{'=' * 72}")
    print(f"Mode:          {args.mode}")
    print(f"Baseline:      {len(baseline_resolved)}/{len(subset)} resolved")
    print(f"Current:       {len(current_resolved)}/{len(subset)} resolved")
    print(f"Net change:    {net_change:+d}")
    print(f"Regressions:   {regression_count} (limit={args.max_regression})")
    print(f"Improvements:  {improvement_count}")
    print(f"Cost ratio:    {cost_ratio:.3f} (reported only; not used for gate)")
    if target_results["metrics"]:
        print("Target metrics:")
        for metric in target_results["metrics"]:
            print(
                f"  - {metric['metric']}: baseline={metric['baseline']:.4f} "
                f"current={metric['current']:.4f} delta={metric['delta']:.4f} "
                f"({'improved' if metric['improved'] else 'not improved'})"
            )
    if failure_reasons:
        print("Failure reasons:")
        for reason in failure_reasons:
            print(f"  - {reason}")
    print(f"{'=' * 72}\n")

    result = {
        "passed": passed,
        "baseline_count": len(baseline_resolved),
        "current_count": len(current_resolved),
        "net_change": net_change,
        "regression_count": regression_count,
        "improvement_count": improvement_count,
        "regressed_ids": regressed_ids,
        "improved_ids": improved_ids,
        "resolved_pass": resolved_pass,
        "target_pass": target_pass,
        "regression_pass": regression_pass,
        "cost_gate_enabled": False,
        "cost_ratio": cost_ratio,
        "failure_reasons": failure_reasons,
        "target_metric_results": target_results,
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "gate_config": {
            "mode": args.mode,
            "max_regression": args.max_regression,
            "min_improvement": args.min_improvement,
            "min_target_improvement": args.min_target_improvement,
            "cost_report_threshold": args.max_cost_ratio,
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Gate result saved to {args.output}")

    print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
