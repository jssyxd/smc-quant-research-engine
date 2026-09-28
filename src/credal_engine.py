"""
Credal Uncertainty Engine for Smart Money Concepts (SMC) Quantitative Trading.

Based on NeurIPS 2025 Credal Transformer (arXiv:2510.12137) and Subjective Logic /
Evidential Deep Learning principles.

Prevents "Artificial Certainty" and trade hallucinations by quantifying epistemic
uncertainty (vacuity) and conflict mass before issuing high-conviction trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import numpy as np


class CredalSignal(str, Enum):
    """Signal classification output from the Credal Uncertainty Engine."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class CredalDecision:
    """Immutable result container for credal uncertainty evaluation.

    Provides direct attribute access, dict-like subscription (`decision['p_bull']`),
    and complete evidential decomposition.
    """
    p_bull: float
    p_bear: float
    p_neutral: float
    uncertainty: float
    should_abstain: bool
    credal_signal: CredalSignal
    confidence: float
    b_bull: float
    b_bear: float
    b_neutral: float
    e_bull: float
    e_bear: float
    e_neutral: float
    alpha_bull: float
    alpha_bear: float
    alpha_neutral: float
    dirichlet_strength: float
    abstain_reason: Optional[str] = None
    indicators: Optional[Dict[str, Any]] = None

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as err:
            raise KeyError(f"Key '{item}' not found in CredalDecision") from err

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a dictionary representation."""
        data = asdict(self)
        data["credal_signal"] = str(self.credal_signal)
        return data

    @property
    def is_bullish(self) -> bool:
        return not self.should_abstain and self.credal_signal == CredalSignal.BULLISH

    @property
    def is_bearish(self) -> bool:
        return not self.should_abstain and self.credal_signal == CredalSignal.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self.credal_signal == CredalSignal.NEUTRAL


# Default factor weights calibrated for SMC high-conviction detection
DEFAULT_BULL_FACTORS: Dict[str, float] = {
    "htf_bull_bias": 2.0,
    "bull_ob": 2.0,
    "bull_fvg": 1.5,
    "ssl_swept": 1.5,
    "in_discount": 1.0,
}

DEFAULT_BEAR_FACTORS: Dict[str, float] = {
    "htf_bear_bias": 2.0,
    "bear_ob": 2.0,
    "bear_fvg": 1.5,
    "bsl_swept": 1.5,
    "in_premium": 1.0,
}
# Aliases to accommodate varied naming conventions across quant repositories
INDICATOR_ALIASES: Dict[str, str] = {
    # Bullish indicators
    "htfbullbias": "htf_bull_bias",
    "htf_bull": "htf_bull_bias",
    "htf_trend_bull": "htf_bull_bias",
    "trend_up": "htf_bull_bias",
    "trendup": "htf_bull_bias",
    "bullobsignal": "bull_ob",
    "bull_ob_signal": "bull_ob",
    "bull_ob": "bull_ob",
    "bullish_ob": "bull_ob",
    "bullishob": "bull_ob",
    "bull_fvg": "bull_fvg",
    "bullfvg": "bull_fvg",
    "bullishob": "bull_ob",
    "sslswept": "ssl_swept",
    "ssl_swept": "ssl_swept",
    "ssl_sweep": "ssl_swept",
    "swept_ssl": "ssl_swept",
    "indiscount": "in_discount",
    "in_discount": "in_discount",
    "discount": "in_discount",

    # Bearish indicators
    "htfbearbias": "htf_bear_bias",
    "htf_bear_bias": "htf_bear_bias",
    "htf_bear": "htf_bear_bias",
    "htf_trend_bear": "htf_bear_bias",
    "trend_down": "htf_bear_bias",
    "trenddown": "htf_bear_bias",
    "bearobsignal": "bear_ob",
    "bear_ob_signal": "bear_ob",
    "bear_ob": "bear_ob",
    "bearish_ob": "bear_ob",
    "bearishob": "bear_ob",
    "bear_fvg": "bear_fvg",
    "bearfvg": "bear_fvg",
    "bslswept": "bsl_swept",
    "bsl_swept": "bsl_swept",
    "bsl_sweep": "bsl_swept",
    "swept_bsl": "bsl_swept",
    "inpremium": "in_premium",
    "in_premium": "in_premium",
    "premium": "in_premium",
}


class CredalSMCEngine:
    """Mathematical Credal Uncertainty Engine for SMC Confluence Scoring.

    Parameters
    ----------
    u_max : float, default=0.35
        Epistemic uncertainty (vacuity) threshold. If vacuity u > u_max,
        the model abstains due to insufficient empirical evidence.
    delta : float, default=0.15
        Belief margin threshold. If |b_bull - b_bear| < delta while evidence
        is present, conflicting evidence trigger fires and the model abstains.
    bull_factors : Optional[Dict[str, float]], default=None
        Weight mapping for bullish SMC factors.
    bear_factors : Optional[Dict[str, float]], default=None
        Weight mapping for bearish SMC factors.
    K : int, default=3
        Number of classes [BULLISH, BEARISH, NEUTRAL].
    min_evidence_conflict : float, default=1e-6
        Tolerance threshold above which evidence is considered non-zero.
    """

    def __init__(
        self,
        u_max: float = 0.35,
        delta: float = 0.15,
        bull_factors: Optional[Dict[str, float]] = None,
        bear_factors: Optional[Dict[str, float]] = None,
        K: int = 3,
        min_evidence_conflict: float = 1e-6,
    ) -> None:
        if u_max <= 0.0 or u_max > 1.0:
            raise ValueError(f"u_max must be in (0, 1], got {u_max}")
        if delta < 0.0 or delta > 1.0:
            raise ValueError(f"delta must be in [0, 1], got {delta}")
        if K < 2:
            raise ValueError(f"K must be at least 2, got {K}")

        self.u_max = float(u_max)
        self.delta = float(delta)
        self.K = int(K)
        self.min_evidence_conflict = float(min_evidence_conflict)

        self.bull_factors = dict(bull_factors if bull_factors is not None else DEFAULT_BULL_FACTORS)
        self.bear_factors = dict(bear_factors if bear_factors is not None else DEFAULT_BEAR_FACTORS)

    def _normalize_key(self, key: str) -> str:
        """Normalize indicator key by stripping whitespace and punctuation."""
        cleaned = key.strip().lower().replace("-", "_").replace(" ", "_")
        return INDICATOR_ALIASES.get(cleaned, INDICATOR_ALIASES.get(cleaned.replace("_", ""), cleaned))

    def extract_evidence(
        self,
        indicators: Optional[Union[Mapping[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> Tuple[float, float, float]:
        """Compute evidence mass (e_bull, e_bear, e_neutral) from indicators.

        Accepts dictionary, mapping, Series-like object, or keyword arguments.
        """
        combined: Dict[str, Any] = {}
        if indicators is not None:
            if hasattr(indicators, "to_dict"):
                combined.update(indicators.to_dict())
            elif isinstance(indicators, Mapping):
                combined.update(indicators)
            elif hasattr(indicators, "__dict__"):
                combined.update(vars(indicators))
        if kwargs:
            combined.update(kwargs)

        # Allow caller to directly provide e_bull / e_bear / e_neutral
        direct_e_bull = combined.get("e_bull") or combined.get("evidence_bull")
        direct_e_bear = combined.get("e_bear") or combined.get("evidence_bear")
        direct_e_neutral = combined.get("e_neutral") or combined.get("evidence_neutral")

        if direct_e_bull is not None or direct_e_bear is not None:
            e_bull = float(direct_e_bull or 0.0)
            e_bear = float(direct_e_bear or 0.0)
            e_neutral = float(direct_e_neutral or 0.0)
            return max(0.0, e_bull), max(0.0, e_bear), max(0.0, e_neutral)

        # Map input keys to normalized canonical keys
        normalized_inputs: Dict[str, float] = {}
        for k, v in combined.items():
            norm_k = self._normalize_key(k)
            # Interpret boolean or numerical factor presence
            val: float = 0.0
            if isinstance(v, (bool, np.bool_)):
                val = 1.0 if v else 0.0
            elif isinstance(v, (int, float, np.number)):
                val = float(v)
            elif v is not None:
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    val = 0.0
            normalized_inputs[norm_k] = max(0.0, val)

        # Accumulate bullish evidence
        e_bull = 0.0
        for factor, weight in self.bull_factors.items():
            factor_val = normalized_inputs.get(factor, 0.0)
            if factor_val > 0.0:
                e_bull += weight * factor_val

        # Accumulate bearish evidence
        e_bear = 0.0
        for factor, weight in self.bear_factors.items():
            factor_val = normalized_inputs.get(factor, 0.0)
            if factor_val > 0.0:
                e_bear += weight * factor_val

        e_neutral = max(0.0, float(normalized_inputs.get("neutral", 0.0)))

        return float(e_bull), float(e_bear), float(e_neutral)

    def compute_from_evidence(
        self,
        e_bull: float,
        e_bear: float,
        e_neutral: float = 0.0,
        raw_indicators: Optional[Dict[str, Any]] = None,
    ) -> CredalDecision:
        """Compute credal uncertainty, beliefs, probabilities, and abstention decision.

        Mathematical formulations:
            alpha_k = e_k + 1
            S = sum_{k=1}^K alpha_k = sum_{k=1}^K e_k + K
            u = K / S (vacuity / epistemic uncertainty)
            b_k = e_k / S (expected belief mass)
            p_k = alpha_k / S = b_k + u / K (Dirichlet mean probability)
        """
        e_bull = max(0.0, float(e_bull))
        e_bear = max(0.0, float(e_bear))
        e_neutral = max(0.0, float(e_neutral))

        # 1. Dirichlet parameters and strength
        alpha_bull = e_bull + 1.0
        alpha_bear = e_bear + 1.0
        alpha_neutral = e_neutral + 1.0
        S = alpha_bull + alpha_bear + alpha_neutral

        # 2. Epistemic uncertainty (vacuity)
        # When all e_k == 0 -> S = K -> u = K/K = 1.0 (total ignorance)
        u = self.K / S

        # 3. Expected beliefs and Dirichlet expected probabilities
        b_bull = e_bull / S
        b_bear = e_bear / S
        b_neutral = e_neutral / S

        p_bull = alpha_bull / S
        p_bear = alpha_bear / S
        p_neutral = alpha_neutral / S

        # 4. Credal Abstention Decision
        # Condition A: Epistemic uncertainty exceeds allowed maximum (vacuity)
        high_uncertainty = u > self.u_max

        # Condition B: Conflicting evidence exists (e_bull and e_bear compete closely)
        total_directional_evidence = e_bull + e_bear
        has_evidence = total_directional_evidence > self.min_evidence_conflict
        belief_diff = abs(b_bull - b_bear)
        conflicting_evidence = has_evidence and (belief_diff < self.delta)

        should_abstain = bool(high_uncertainty or conflicting_evidence)

        # 5. Signal classification & Confidence calculation
        abstain_reason: Optional[str] = None
        if high_uncertainty and conflicting_evidence:
            abstain_reason = "HIGH_UNCERTAINTY_AND_CONFLICT"
        elif high_uncertainty:
            abstain_reason = "HIGH_UNCERTAINTY"
        elif conflicting_evidence:
            abstain_reason = "CONFLICTING_EVIDENCE"

        if should_abstain:
            # Model abstains: returns NEUTRAL with 0.0 confidence
            credal_signal = CredalSignal.NEUTRAL
            confidence = 0.0
        else:
            if b_bull > b_bear:
                credal_signal = CredalSignal.BULLISH
                confidence = float(b_bull - b_bear)
            elif b_bear > b_bull:
                credal_signal = CredalSignal.BEARISH
                confidence = float(b_bear - b_bull)
            else:
                credal_signal = CredalSignal.NEUTRAL
                confidence = 0.0

        return CredalDecision(
            p_bull=float(p_bull),
            p_bear=float(p_bear),
            p_neutral=float(p_neutral),
            uncertainty=float(u),
            should_abstain=should_abstain,
            credal_signal=credal_signal,
            confidence=round(confidence, 6),
            b_bull=float(b_bull),
            b_bear=float(b_bear),
            b_neutral=float(b_neutral),
            e_bull=float(e_bull),
            e_bear=float(e_bear),
            e_neutral=float(e_neutral),
            alpha_bull=float(alpha_bull),
            alpha_bear=float(alpha_bear),
            alpha_neutral=float(alpha_neutral),
            dirichlet_strength=float(S),
            abstain_reason=abstain_reason,
            indicators=raw_indicators,
        )

    def evaluate(
        self,
        indicators: Optional[Union[Mapping[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> CredalDecision:
        """Evaluate a bar's SMC indicators and compute the Credal decision.

        Examples
        --------
        >>> engine = CredalSMCEngine()
        >>> decision = engine.evaluate(
        ...     htf_bull_bias=True,
        ...     bull_ob=True,
        ...     ssl_swept=True,
        ...     in_discount=True
        ... )
        >>> decision.should_abstain
        False
        >>> decision.credal_signal
        <CredalSignal.BULLISH: 'BULLISH'>
        """
        e_bull, e_bear, e_neutral = self.extract_evidence(indicators=indicators, **kwargs)

        # Keep original indicators dictionary for transparency if available
        raw_dict: Dict[str, Any] = {}
        if indicators is not None:
            if hasattr(indicators, "to_dict"):
                raw_dict.update(indicators.to_dict())
            elif isinstance(indicators, Mapping):
                raw_dict.update(dict(indicators))
        if kwargs:
            raw_dict.update(kwargs)

        return self.compute_from_evidence(
            e_bull=e_bull,
            e_bear=e_bear,
            e_neutral=e_neutral,
            raw_indicators=raw_dict if raw_dict else None,
        )

    def __call__(
        self,
        indicators: Optional[Union[Mapping[str, Any], Any]] = None,
        **kwargs: Any,
    ) -> CredalDecision:
        """Allow engine to be called as a function: engine(indicators)."""
        return self.evaluate(indicators=indicators, **kwargs)

    def evaluate_df(self, df: Any) -> Any:
        """Evaluate a pandas DataFrame containing SMC indicator columns.

        Returns a new DataFrame containing credal metrics and trade signals.
        """
        results = [self.evaluate(row.to_dict()) for _, row in df.iterrows()]
        import pandas as pd

        res_df = pd.DataFrame([
            {
                "p_bull": r.p_bull,
                "p_bear": r.p_bear,
                "p_neutral": r.p_neutral,
                "uncertainty": r.uncertainty,
                "should_abstain": r.should_abstain,
                "credal_signal": str(r.credal_signal),
                "confidence": r.confidence,
                "abstain_reason": r.abstain_reason or "",
                "b_bull": r.b_bull,
                "b_bear": r.b_bear,
                "e_bull": r.e_bull,
                "e_bear": r.e_bear,
            }
            for r in results
        ], index=df.index)
        return res_df
