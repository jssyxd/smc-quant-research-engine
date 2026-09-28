import sys
import json
import re
from pathlib import Path

def validate_deliverable(report_path: str) -> bool:
    """
    Enforces the Inviolable Dual-Convergence Protocol on Agent research deliverables.
    An agent's report is REJECTED (returns False) unless it starts with one of the two explicit conclusions.
    """
    p = Path(report_path)
    if not p.exists():
        print(f"ERROR: Deliverable file {report_path} does not exist.")
        return False

    content = p.read_text(encoding="utf-8")

    # Hard-coded dual convergence signatures
    has_conclusion_1 = (
        "【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】" in content
        or "CONCLUSION 1: NO RESEARCH VALUE" in content
        or "[STATUS: ABANDONED]" in content
    )
    has_conclusion_2 = (
        "【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】" in content
        or "CONCLUSION 2: PRODUCTION READY" in content
        or "[STATUS: PRODUCTION_READY]" in content
    )

    if has_conclusion_1 and has_conclusion_2:
        print("REJECTED: Ambiguous deliverable containing both conclusions. Agent must converge to exactly ONE final outcome.")
        return False

    if not (has_conclusion_1 or has_conclusion_2):
        print("REJECTED: Deliverable lacks mandatory Dual-Convergence Termination Header.")
        print("Agent MUST explicitly declare either:")
        print("1. 【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】")
        print("2. 【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】")
        return False

    # Check for prohibited soft language
    prohibited_phrases = [
        "需要进一步观察",
        "建议未来继续测试",
        "可能在某些未见市场有效",
        "inconclusive",
        "requires further observation",
    ]
    for phrase in prohibited_phrases:
        if phrase in content:
            print(f"REJECTED: Deliverable contains prohibited ambiguous phrase: '{phrase}'. Agent must converge definitively.")
            return False

    print("VERIFIED: Deliverable strictly adheres to the Dual-Convergence Protocol.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/validate_agent_convergence.py <path_to_report.md>")
        sys.exit(1)
    ok = validate_deliverable(sys.argv[1])
    sys.exit(0 if ok else 1)
