#!/usr/bin/env python3
"""
Interactive / Autonomous Agent Workflow CLI Entrypoint.
Guides any Agent or Quant Researcher through the strict 5-phase iterative cycle
and enforces convergence to Conclusion 1 or Conclusion 2.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_agent_convergence_lifecycle import run_agent_workflow_cycle
from scripts.validate_agent_convergence import validate_deliverable

def main():
    parser = argparse.ArgumentParser(description="Autonomous Quant Agent Dual-Convergence CLI")
    parser.add_argument("--strategy", type=str, default="SMC_Optimized_V1_V3", help="Name of strategy to evaluate")
    parser.add_argument("--output", type=str, default="docs/research/WORKFLOW_AGENT_FINAL_VERDICT.md", help="Path for research deliverable")
    parser.add_argument("--validate-only", type=str, default=None, help="Validate an existing deliverable path")

    args = parser.parse_args()

    if args.validate_only:
        print(f">>> Validating deliverable: {args.validate_only}")
        ok = validate_deliverable(args.validate_only)
        sys.exit(0 if ok else 1)

    print("================================================================================")
    print(f"  QUANT AI AGENT SWARM: EXECUTING CONVERGENCE LIFECYCLE [{args.strategy}]")
    print("================================================================================")

    res = run_agent_workflow_cycle(args.strategy, args.output)
    print("\n>>> Lifecycle Execution Finished.")
    print(f">>> Verdict: {res['verdict']}")
    print(f">>> Final State: {res['final_state']}")
    print(f">>> Report: {res['report_path']}")

    # Self-validation check
    print("\n>>> Running Gate Verification on Generated Deliverable...")
    valid = validate_deliverable(res['report_path'])
    if not valid:
        print("[!] FATAL: Self-validation failed. Report does not meet Dual-Convergence Protocol.")
        sys.exit(1)

    print(">>> SUCCESS: Deliverable 100% compliant with Dual-Convergence Protocol.")
    sys.exit(0)

if __name__ == "__main__":
    main()
