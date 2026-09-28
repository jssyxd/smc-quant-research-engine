import os
import sys
import unittest
from pathlib import Path

# Fix sys.path for root imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_convergence_state_machine import (
    AgentConvergenceStateMachine,
    ConvergenceState,
    CONCLUSION_1_TEXT,
    CONCLUSION_2_TEXT,
)

class TestAgentConvergenceStateMachine(unittest.TestCase):
    def test_clean_production_ready_convergence(self):
        sm = AgentConvergenceStateMachine("SMC_Production_Strategy")
        report = sm.evaluate_termination(
            has_positive_expectancy_under_friction=True,
            survives_pessimistic_penetration=True,
            survives_intra_bar_pessimism=True,
            survives_shared_capital_reserve=True,
            zero_cherry_picking_passed=True,
            ex_ante_constraint_monotonic=True,
            whitehat_invariants_100pct=True,
        )
        self.assertEqual(report.final_state, ConvergenceState.TERMINAL_PRODUCTION_READY)
        self.assertEqual(report.verdict, CONCLUSION_2_TEXT)
        self.assertEqual(len(report.falsification_reasons), 0)

    def test_falsification_convergence_when_cherry_picking(self):
        sm = AgentConvergenceStateMachine("SMC_Cherry_Picked")
        report = sm.evaluate_termination(
            has_positive_expectancy_under_friction=True,
            survives_pessimistic_penetration=True,
            survives_intra_bar_pessimism=True,
            survives_shared_capital_reserve=True,
            zero_cherry_picking_passed=False, # Failed zero cherry-picking
            ex_ante_constraint_monotonic=True,
            whitehat_invariants_100pct=True,
        )
        self.assertEqual(report.final_state, ConvergenceState.TERMINAL_NO_RESEARCH_VALUE)
        self.assertEqual(report.verdict, CONCLUSION_1_TEXT)
        self.assertIn("选择偏差: 依赖事后挑选优势币种，在全量适用标的横截面上整体崩溃。", report.falsification_reasons)

    def test_falsification_convergence_when_friction_eats_pnl(self):
        sm = AgentConvergenceStateMachine("High_Frequency_Friction_Strategy")
        report = sm.evaluate_termination(
            has_positive_expectancy_under_friction=False, # Eaten by fees
            survives_pessimistic_penetration=False,
            survives_intra_bar_pessimism=False,
            survives_shared_capital_reserve=False,
            zero_cherry_picking_passed=True,
            ex_ante_constraint_monotonic=False,
            whitehat_invariants_100pct=True,
        )
        self.assertEqual(report.final_state, ConvergenceState.TERMINAL_NO_RESEARCH_VALUE)
        self.assertEqual(report.verdict, CONCLUSION_1_TEXT)
        self.assertTrue(len(report.falsification_reasons) > 1)

if __name__ == "__main__":
    unittest.main()
