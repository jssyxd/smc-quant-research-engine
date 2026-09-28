import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_agent_convergence import validate_deliverable

class TestAgentConvergenceValidator(unittest.TestCase):
    def test_conclusion_1_accepted(self):
        tmp = Path("tests/test_report_concl1.md")
        tmp.write_text("# Research Report\n\n【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】\n\n- 摩擦过大", encoding="utf-8")
        try:
            self.assertTrue(validate_deliverable(str(tmp)))
        finally:
            tmp.unlink(missing_ok=True)

    def test_conclusion_2_accepted(self):
        tmp = Path("tests/test_report_concl2.md")
        tmp.write_text("# Research Report\n\n【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】\n\n- 全量标的无偏检验通过", encoding="utf-8")
        try:
            self.assertTrue(validate_deliverable(str(tmp)))
        finally:
            tmp.unlink(missing_ok=True)

    def test_ambiguous_report_rejected(self):
        tmp = Path("tests/test_report_ambiguous.md")
        tmp.write_text("# Research Report\n\n【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】\n\n需要进一步观察未来表现", encoding="utf-8")
        try:
            self.assertFalse(validate_deliverable(str(tmp)))
        finally:
            tmp.unlink(missing_ok=True)

    def test_missing_header_rejected(self):
        tmp = Path("tests/test_report_missing.md")
        tmp.write_text("# Research Report\n\n回测表现一般，建议继续微调参数。", encoding="utf-8")
        try:
            self.assertFalse(validate_deliverable(str(tmp)))
        finally:
            tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
