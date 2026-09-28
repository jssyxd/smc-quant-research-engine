"""
Agent Multi-Round Iterative State Machine & Convergence Harness.

Enforces that any autonomous quant agent workflow iteratively transitions
through strict verification gates and terminates at EXACTLY ONE of two final states:
1. State 1 (TERMINAL_ABANDONED):
   "该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一。"
2. State 2 (TERMINAL_PRODUCTION_READY):
   "该策略在严格避免过拟合的情况下能够合理优化提升 PnL。"
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

class ConvergenceState(str, Enum):
    INITIALIZED = "INITIALIZED"
    PHASE1_SCOPE_SIZING_AUDITED = "PHASE1_SCOPE_SIZING_AUDITED"
    PHASE2_DUAL_CAPITAL_EVALUATED = "PHASE2_DUAL_CAPITAL_EVALUATED"
    PHASE3_PESSIMISTIC_MATCHING_AUDITED = "PHASE3_PESSIMISTIC_MATCHING_AUDITED"
    PHASE4_ANTI_OVERFIT_VERIFIED = "PHASE4_ANTI_OVERFIT_VERIFIED"
    TERMINAL_NO_RESEARCH_VALUE = "TERMINAL_NO_RESEARCH_VALUE"
    TERMINAL_PRODUCTION_READY = "TERMINAL_PRODUCTION_READY"

CONCLUSION_1_TEXT = "【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】"
CONCLUSION_2_TEXT = "【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】"

@dataclass
class ConvergenceReport:
    final_state: ConvergenceState
    verdict: str
    checklist: Dict[str, bool]
    metrics: Dict[str, Any]
    falsification_reasons: List[str]

class AgentConvergenceStateMachine:
    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        self.state = ConvergenceState.INITIALIZED
        self.history: List[str] = []

    def evaluate_termination(
        self,
        has_positive_expectancy_under_friction: bool,
        survives_pessimistic_penetration: bool,
        survives_intra_bar_pessimism: bool,
        survives_shared_capital_reserve: bool,
        zero_cherry_picking_passed: bool,
        ex_ante_constraint_monotonic: bool,
        whitehat_invariants_100pct: bool,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ConvergenceReport:
        """
        Evaluates the dual convergence outcome with zero tolerance for ambiguity.
        """
        metrics = metrics or {}
        checklist = {
            "has_positive_expectancy_under_friction": has_positive_expectancy_under_friction,
            "survives_pessimistic_penetration": survives_pessimistic_penetration,
            "survives_intra_bar_pessimism": survives_intra_bar_pessimism,
            "survives_shared_capital_reserve": survives_shared_capital_reserve,
            "zero_cherry_picking_passed": zero_cherry_picking_passed,
            "ex_ante_constraint_monotonic": ex_ante_constraint_monotonic,
            "whitehat_invariants_100pct": whitehat_invariants_100pct,
        }

        falsification_reasons = []
        if not has_positive_expectancy_under_friction:
            falsification_reasons.append("交易摩擦吞噬: 扣除 Maker 0.02% / Taker 0.05% 及滑点后无正期望。")
        if not survives_pessimistic_penetration:
            falsification_reasons.append("依赖排队成交幻觉: 挂单必须严格穿透盘口，穿透后胜率与收益断崖暴跌。")
        if not survives_intra_bar_pessimism:
            falsification_reasons.append("依赖同Bar假止盈红利: 同根K线同时触碰TP和SL时，先止损出局导致利润回吐殆尽。")
        if not survives_shared_capital_reserve:
            falsification_reasons.append("资金冲突与相关性共振: 共享资金池(限2仓并发+留50%现金)下，多币同动暴跌吞噬利润。")
        if not zero_cherry_picking_passed:
            falsification_reasons.append("选择偏差: 依赖事后挑选优势币种，在全量适用标的横截面上整体崩溃。")
        if not ex_ante_constraint_monotonic:
            falsification_reasons.append("换皮过拟合: 理论约束在样本外盲测中未能呈现单调减亏增益效果。")
        if not whitehat_invariants_100pct:
            falsification_reasons.append("因果与守恒破裂: 存在未来前瞻数据泄漏或复式记账资金不平衡。")

        # Convergence Rule:
        # All conditions must be strictly satisfied for Conclusion 2.
        # If any condition fails, the strategy unconditionally converges to Conclusion 1.
        all_passed = all(checklist.values())
        if all_passed:
            self.state = ConvergenceState.TERMINAL_PRODUCTION_READY
            verdict = CONCLUSION_2_TEXT
        else:
            self.state = ConvergenceState.TERMINAL_NO_RESEARCH_VALUE
            verdict = CONCLUSION_1_TEXT

        return ConvergenceReport(
            final_state=self.state,
            verdict=verdict,
            checklist=checklist,
            metrics=metrics,
            falsification_reasons=falsification_reasons,
        )
