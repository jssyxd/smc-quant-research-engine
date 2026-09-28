"""
Statistical Anti-Overfitting and Backtest Audit Engine.

Implements the gold standard quantitative statistical tests addressing
Alpha Hallucination, selection bias under multiple testing, and false discovery
rates (FDR) in automated strategy generation and backtesting:

1. Deflated Sharpe Ratio (DSR) (Bailey & López de Prado, 2014):
   Adjusts observed Sharpe ratio for track record length, non-normality
   (skewness, kurtosis), number of trials, and variance across trials.

2. Probability of Backtest Overfitting (PBO) via Combinatorial Symmetric
   Cross-Validation (CSCV) (Bailey, Borwein, López de Prado, Zhu, 2015):
   Measures probability that the strategy with best In-Sample rank ranks
   below the median Out-of-Sample.

3. Haircut Sharpe Ratio (Harvey & Liu, 2014/2015):
   Penalizes annualized Sharpe ratio by inverting multiple-testing adjusted
   p-values (Bonferroni, BHY, Holm, correlation-adjusted).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# Euler-Mascheroni constant (gamma)
EULER_MASCHERONI: float = 0.57721566490153286060


# ---------------------------------------------------------------------------
# Statistical & Numerical Utilities
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_ppf(p: float) -> float:
    """
    Inverse normal cumulative distribution function (quantile/probit function).
    
    Uses Peter J. Acklam's rational approximation with relative error < 1.15e-9.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def log_norm_tail(t: float) -> float:
    """Accurate log(1 - Phi(t)) for t >= 0, preventing underflow in extreme tails."""
    if t < 0.0:
        return math.log(max(1e-300, 1.0 - norm_cdf(t)))
    if t < 7.0:
        p = 1.0 - norm_cdf(t)
        return math.log(max(p, 1e-300))
    # Asymptotic tail expansion: 1 - Phi(t) ~ phi(t)/t * (1 - 1/t^2 + 3/t^4)
    log_phi = -0.5 * math.log(2.0 * math.pi) - 0.5 * t * t
    corr = max(1e-12, 1.0 - 1.0 / (t * t) + 3.0 / (t * t * t * t))
    return log_phi - math.log(t) + math.log(corr)


def invert_log_p(log_p: float) -> float:
    """Find t >= 0 such that log(1 - Phi(t)) = log_p."""
    if log_p >= math.log(0.5):
        return norm_ppf(1.0 - math.exp(log_p))
    if log_p > -35.0:  # t < ~8.5
        return norm_ppf(1.0 - math.exp(log_p))
    # Newton-Raphson on asymptotic approximation
    t = math.sqrt(-2.0 * log_p)
    for _ in range(12):
        f = -0.5 * t * t - math.log(t) - 0.5 * math.log(2.0 * math.pi) - log_p
        df = -t - 1.0 / t
        step = f / df
        t -= step
        if abs(step) < 1e-10:
            break
    return max(0.0, t)


def expected_maximum_sr(n_trials: int, var_trials: Optional[float] = None) -> float:
    """
    Computes expected maximum Sharpe ratio under null hypothesis of no skill:
    
        E[max_N] ≈ sqrt(2 ln N) + gamma / sqrt(2 ln N)
    
    scaled by sqrt(V[SR_n]) if trial variance is provided:
        E[max_N] = sqrt(V[SR_n]) * (sqrt(2 ln N) + gamma / sqrt(2 ln N))
    """
    if n_trials <= 1:
        return 0.0
    
    log_n = math.log(n_trials)
    sqrt_2_log_n = math.sqrt(2.0 * log_n)
    
    # Asymptotic EVT expected maximum for standard normal
    e_max_std = sqrt_2_log_n + (EULER_MASCHERONI / sqrt_2_log_n)
    
    if var_trials is None:
        return e_max_std
    
    if var_trials <= 0.0:
        return 0.0
        
    return math.sqrt(var_trials) * e_max_std


# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DSRResult:
    """Result container for Deflated Sharpe Ratio calculation."""
    dsr: float
    p_value: float
    expected_max_sr: float
    z_stat: float
    std_error: float
    observed_sr: float
    n_trials: int
    var_trials: float
    passes: bool

    @property
    def is_significant(self) -> bool:
        """True if DSR is statistically significant at 95% confidence level."""
        return self.passes

    def __float__(self) -> float:
        return float(self.dsr)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    
    Computes probability that an observed Sharpe ratio is false discovery
    under multiple testing and non-normal return distributions.
    """

    def __init__(
        self,
        sr: Optional[float] = None,
        t: Optional[int] = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        n_trials: int = 1,
        var_trials: Optional[float] = None,
        trials: Optional[Sequence[float]] = None,
        returns: Optional[Sequence[float]] = None,
        significance_level: float = 0.05,
    ) -> None:
        self.significance_level = significance_level
        self.result = self.compute(
            sr=sr,
            t=t,
            skew=skew,
            kurtosis=kurtosis,
            n_trials=n_trials,
            var_trials=var_trials,
            trials=trials,
            returns=returns,
            significance_level=significance_level,
        )

    @classmethod
    def compute(
        cls,
        sr: Optional[float] = None,
        t: Optional[int] = None,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        n_trials: int = 1,
        var_trials: Optional[float] = None,
        trials: Optional[Sequence[float]] = None,
        returns: Optional[Sequence[float]] = None,
        significance_level: float = 0.05,
    ) -> DSRResult:
        """
        Compute Deflated Sharpe Ratio (DSR) and its p-value.
        
        Parameters:
        -----------
        sr : float, optional
            Observed Sharpe ratio. If returns is provided, computed directly.
        t : int, optional
            Track record length (number of observations). If returns provided, len(returns).
        skew : float, default 0.0
            Skewness of returns (gamma_3).
        kurtosis : float, default 3.0
            Kurtosis of returns (gamma_4). If < 1.0, assumed to be excess kurtosis and +3 added.
        n_trials : int, default 1
            Number of strategy trials tested (N).
        var_trials : float, optional
            Variance of trial Sharpe ratios V[SR_n].
        trials : Sequence[float], optional
            Array of trial Sharpe ratios to automatically compute n_trials and var_trials.
        returns : Sequence[float], optional
            1D array of return observations. Automatically computes sr, t, skew, kurtosis.
        significance_level : float, default 0.05
            Significance threshold (alpha). Strategy passes if p_value <= significance_level
            (equivalently DSR >= 1 - significance_level).
        """
        # If returns sequence is passed, compute sample moments
        if returns is not None:
            ret_arr = np.asarray(returns, dtype=np.float64)
            t = len(ret_arr)
            if t < 3:
                raise ValueError("Returns series must contain at least 3 observations.")
            mean = float(np.mean(ret_arr))
            std = float(np.std(ret_arr, ddof=1))
            if std == 0.0:
                raise ValueError("Returns standard deviation is zero.")
            sr = mean / std
            
            # Moments
            diff = ret_arr - mean
            m2 = float(np.mean(diff ** 2))
            m3 = float(np.mean(diff ** 3))
            m4 = float(np.mean(diff ** 4))
            skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
            kurtosis = m4 / (m2 ** 2) if m2 > 0 else 3.0

        if sr is None:
            raise ValueError("Must provide either observed Sharpe ratio 'sr' or 'returns'.")
        if t is None or t <= 1:
            raise ValueError("Track record length 't' must be greater than 1.")

        # If trial Sharpe ratios passed, deduce n_trials and var_trials
        if trials is not None:
            trial_arr = np.asarray(trials, dtype=np.float64)
            n_trials = len(trial_arr)
            if n_trials > 1:
                var_trials = float(np.var(trial_arr, ddof=1))
            else:
                var_trials = 0.0

        # Normalise kurtosis if excess kurtosis was supplied (kurtosis < 1)
        kurt_val = kurtosis if kurtosis >= 1.0 else (kurtosis + 3.0)

        # Expected maximum Sharpe ratio under H0 of no skill
        e_max = expected_maximum_sr(n_trials=n_trials, var_trials=var_trials)

        # Mertens (2002) / Lo (2002) standard error of Sharpe ratio
        # se = sqrt( (1 - skew * SR + (kurt - 1) / 4 * SR^2) / (T - 1) )
        numerator = 1.0 - skew * sr + ((kurt_val - 1.0) / 4.0) * (sr ** 2)
        numerator = max(1e-12, numerator)
        std_error = math.sqrt(numerator / (t - 1))

        # Standardized z-statistic against deflated null benchmark
        z_stat = (sr - e_max) / std_error

        # DSR metric is cumulative normal probability Phi(z)
        dsr = norm_cdf(z_stat)

        # DSR p-value: P(SR <= E[max_N]) under observed sample distribution
        # Equivalently: probability that H0 produces a Sharpe ratio >= observed SR
        p_value = norm_cdf(-z_stat)

        passes = (p_value <= significance_level) and (dsr >= (1.0 - significance_level))

        return DSRResult(
            dsr=dsr,
            p_value=p_value,
            expected_max_sr=e_max,
            z_stat=z_stat,
            std_error=std_error,
            observed_sr=float(sr),
            n_trials=int(n_trials),
            var_trials=float(var_trials if var_trials is not None else 1.0),
            passes=passes,
        )

    # Delegate attribute lookups and float conversion to the underlying DSRResult
    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    def __float__(self) -> float:
        return float(self.result.dsr)

    def __repr__(self) -> str:
        return (
            f"DeflatedSharpeRatio(dsr={self.result.dsr:.4f}, "
            f"p_value={self.result.p_value:.4e}, "
            f"expected_max_sr={self.result.expected_max_sr:.4f}, "
            f"passes={self.result.passes})"
        )


def deflated_sharpe_ratio(
    sr: Optional[float] = None,
    t: Optional[int] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int = 1,
    var_trials: Optional[float] = None,
    trials: Optional[Sequence[float]] = None,
    returns: Optional[Sequence[float]] = None,
    significance_level: float = 0.05,
) -> DSRResult:
    """Functional wrapper for Deflated Sharpe Ratio calculation."""
    return DeflatedSharpeRatio.compute(
        sr=sr,
        t=t,
        skew=skew,
        kurtosis=kurtosis,
        n_trials=n_trials,
        var_trials=var_trials,
        trials=trials,
        returns=returns,
        significance_level=significance_level,
    )


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting (PBO) via CSCV
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PBOResult:
    """Result container for Probability of Backtest Overfitting calculation."""
    pbo: float
    prob_loss: float
    degradation: float
    mean_is_perf: float
    mean_oos_perf: float
    n_partitions: int
    n_combinations: int
    logits: np.ndarray
    rank_distribution: np.ndarray
    is_overfitted: bool
    passes: bool

    def __float__(self) -> float:
        return float(self.pbo)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["logits"] = self.logits.tolist()
        d["rank_distribution"] = self.rank_distribution.tolist()
        return d


class ProbabilityOfBacktestOverfitting:
    """
    Combinatorial Symmetric Cross-Validation (CSCV) Engine for PBO.
    
    Reference: Bailey, Borwein, López de Prado, Zhu (2015).
    """

    def __init__(
        self,
        returns_matrix: np.ndarray,
        n_partitions: int = 8,
        metric: str = "sharpe",
        max_combinations: Optional[int] = None,
        pbo_threshold: float = 0.40,
    ) -> None:
        self.pbo_threshold = pbo_threshold
        self.result = self.compute(
            returns_matrix=returns_matrix,
            n_partitions=n_partitions,
            metric=metric,
            max_combinations=max_combinations,
            pbo_threshold=pbo_threshold,
        )

    @classmethod
    def compute(
        cls,
        returns_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
        n_partitions: int = 8,
        metric: str = "sharpe",
        max_combinations: Optional[int] = None,
        pbo_threshold: float = 0.40,
    ) -> PBOResult:
        """
        Compute Probability of Backtest Overfitting (PBO) using CSCV.
        
        Parameters:
        -----------
        returns_matrix : array-like of shape (T, N)
            Matrix of strategy returns across T periods (rows) for N strategies (columns).
        n_partitions : int, default 8
            Number of equal slices (S). Must be an even positive integer (e.g. 6, 8, 10, 16).
        metric : str, default 'sharpe'
            Performance evaluation function: 'sharpe', 'return' (mean), or 'calmar'.
        max_combinations : int, optional
            Limit combinations if C(S, S/2) is large.
        pbo_threshold : float, default 0.40
            Threshold above which strategy search is classified as overfitted.
        """
        matrix = np.asarray(returns_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"returns_matrix must be 2D of shape (T, N), got shape {matrix.shape}")
        
        T, N = matrix.shape
        if T < n_partitions * 2:
            raise ValueError(f"Sample size T={T} too small for n_partitions={n_partitions}.")
        if N < 2:
            raise ValueError(f"Must have at least N=2 strategy trials, got N={N}.")
        if n_partitions % 2 != 0 or n_partitions <= 0:
            raise ValueError(f"n_partitions must be a positive even integer, got {n_partitions}.")

        # Partition T periods into S contiguous slices
        splits = np.array_split(np.arange(T), n_partitions)
        k = n_partitions // 2
        
        # All combinations of S choose S/2
        all_combos = list(combinations(range(n_partitions), k))
        if max_combinations is not None and len(all_combos) > max_combinations:
            step = len(all_combos) / max_combinations
            combos = [all_combos[int(i * step)] for i in range(max_combinations)]
        else:
            combos = all_combos

        def calc_metric(sub_matrix: np.ndarray) -> np.ndarray:
            """Vectorized column-wise performance metric."""
            if metric == "sharpe":
                m = np.mean(sub_matrix, axis=0)
                s = np.std(sub_matrix, axis=0, ddof=1)
                s = np.where(s <= 1e-12, 1e-12, s)
                return m / s
            elif metric == "return":
                return np.mean(sub_matrix, axis=0)
            elif metric == "calmar":
                cum = np.cumprod(1.0 + sub_matrix, axis=0)
                peak = np.maximum.accumulate(cum, axis=0)
                dd = (peak - cum) / peak
                mdd = np.max(dd, axis=0)
                mdd = np.where(mdd <= 1e-12, 1e-12, mdd)
                return (cum[-1] - 1.0) / mdd
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        overfitted_count = 0
        loss_count = 0
        rel_ranks: List[float] = []
        is_best_perfs: List[float] = []
        oos_best_perfs: List[float] = []

        all_blocks = set(range(n_partitions))

        for is_blocks in combos:
            is_blocks_set = set(is_blocks)
            oos_blocks = [b for b in all_blocks if b not in is_blocks_set]

            is_idx = np.concatenate([splits[b] for b in is_blocks])
            oos_idx = np.concatenate([splits[b] for b in oos_blocks])

            is_perf = calc_metric(matrix[is_idx, :])
            oos_perf = calc_metric(matrix[oos_idx, :])

            # Strategy n* with best In-Sample performance
            n_star = int(np.argmax(is_perf))
            is_best_perf = float(is_perf[n_star])
            oos_best_perf = float(oos_perf[n_star])

            is_best_perfs.append(is_best_perf)
            oos_best_perfs.append(oos_best_perf)

            # Rank of n* in OOS: 1 = worst, N = best
            oos_rank = int(np.sum(oos_perf <= oos_best_perf))
            
            # Relative rank percentile lambda_c in (0, 1)
            rel_rank = oos_rank / (N + 1.0)
            rel_ranks.append(rel_rank)

            # Ranks below median Out-of-Sample
            if rel_rank <= 0.5:
                overfitted_count += 1

            if oos_best_perf <= 0.0:
                loss_count += 1

        n_combos = len(combos)
        pbo = overfitted_count / n_combos
        prob_loss = loss_count / n_combos

        ranks_arr = np.array(rel_ranks, dtype=np.float64)
        clipped_ranks = np.clip(ranks_arr, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped_ranks / (1.0 - clipped_ranks))

        mean_is = float(np.mean(is_best_perfs))
        mean_oos = float(np.mean(oos_best_perfs))
        degradation = float(1.0 - (mean_oos / mean_is)) if mean_is != 0.0 else 0.0

        is_overfitted = (pbo >= pbo_threshold) or (prob_loss >= 0.50)
        passes = not is_overfitted

        return PBOResult(
            pbo=pbo,
            prob_loss=prob_loss,
            degradation=degradation,
            mean_is_perf=mean_is,
            mean_oos_perf=mean_oos,
            n_partitions=n_partitions,
            n_combinations=n_combos,
            logits=logits,
            rank_distribution=ranks_arr,
            is_overfitted=is_overfitted,
            passes=passes,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    def __float__(self) -> float:
        return float(self.result.pbo)

    def __repr__(self) -> str:
        return (
            f"ProbabilityOfBacktestOverfitting(pbo={self.result.pbo:.4f}, "
            f"prob_loss={self.result.prob_loss:.4f}, "
            f"degradation={self.result.degradation:.4f}, "
            f"is_overfitted={self.result.is_overfitted})"
        )


def probability_of_backtest_overfitting(
    returns_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
    n_partitions: int = 8,
    metric: str = "sharpe",
    max_combinations: Optional[int] = None,
    pbo_threshold: float = 0.40,
) -> PBOResult:
    """Functional wrapper for PBO computation."""
    return ProbabilityOfBacktestOverfitting.compute(
        returns_matrix=returns_matrix,
        n_partitions=n_partitions,
        metric=metric,
        max_combinations=max_combinations,
        pbo_threshold=pbo_threshold,
    )


# ---------------------------------------------------------------------------
# 3. Haircut Sharpe Ratio (Harvey & Liu)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HaircutResult:
    """Result container for Haircut Sharpe Ratio calculation."""
    haircut_sr: float
    haircut_pct: float
    original_sr: float
    p_unadj: float
    p_adj: float
    n_trials: int
    n_eff: float
    method: str
    passes: bool

    @property
    def is_significant(self) -> bool:
        return self.passes

    def __float__(self) -> float:
        return float(self.haircut_sr)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def haircut_sharpe(
    sr: float,
    n_trials: int,
    t: Union[int, float] = 252,
    method: str = "bonferroni",
    correlation: float = 0.0,
    significance_level: float = 0.05,
    annualized: bool = True,
) -> HaircutResult:
    """
    Harvey & Liu (2014/2015) Haircut Sharpe Ratio.
    
    Penalizes the observed annualized Sharpe ratio for the number of tested
    variations by calculating the multiple-testing adjusted p-value and inverting
    it back to the equivalent unadjusted Sharpe ratio.
    
    Parameters:
    -----------
    sr : float
        Observed annualized Sharpe ratio.
    n_trials : int
        Total number of strategy trials or parameter combinations evaluated.
    t : int or float, default 252
        Sample size (periods or years). If annualized=True and t >= 30 (e.g. 252 or 1000 days),
        automatically scaled to years via t / 252.0.
    method : str, default 'bonferroni'
        Multiple testing correction method:
        - 'bonferroni': Standard family-wise error rate control.
        - 'bhy': Benjamini-Hochberg-Yekutieli controlling False Discovery Rate (FDR).
        - 'holm': Holm step-down correction for best rank.
        - 'exponential': Empirical exponential penalty model from Harvey & Liu.
    correlation : float, default 0.0
        Average pairwise correlation between strategy trials rho in [0, 1).
    significance_level : float, default 0.05
        Significance cutoff.
    annualized : bool, default True
        Whether sr is an annualized Sharpe ratio.
    """
    if sr <= 0.0:
        return HaircutResult(
            haircut_sr=0.0,
            haircut_pct=1.0,
            original_sr=float(sr),
            p_unadj=1.0,
            p_adj=1.0,
            n_trials=int(n_trials),
            n_eff=1.0,
            method=method,
            passes=False,
        )

    if n_trials <= 1:
        # No multiple testing penalty for a single prior hypothesis
        t_stat = sr * math.sqrt(t)
        log_p = log_norm_tail(t_stat)
        p_unadj = min(1.0, 2.0 * math.exp(log_p)) if log_p > -30.0 else 0.0
        return HaircutResult(
            haircut_sr=float(sr),
            haircut_pct=0.0,
            original_sr=float(sr),
            p_unadj=p_unadj,
            p_adj=p_unadj,
            n_trials=1,
            n_eff=1.0,
            method=method,
            passes=p_unadj <= significance_level,
        )

    # Effective number of independent trials accounting for correlation
    rho = max(0.0, min(0.9999, correlation))
    n_eff = 1.0 + (n_trials - 1.0) * (1.0 - rho)

    # Empirical exponential model option
    if method == "exponential":
        # Empirical rule-of-thumb penalty: haircut = 1 - exp(-0.15 * log(n_eff))
        haircut_pct = min(0.99, max(0.0, 1.0 - math.exp(-0.18 * math.log(n_eff))))
        adj_sr = max(0.0, sr * (1.0 - haircut_pct))
        return HaircutResult(
            haircut_sr=adj_sr,
            haircut_pct=haircut_pct,
            original_sr=float(sr),
            p_unadj=0.0,
            p_adj=0.0,
            n_trials=int(n_trials),
            n_eff=n_eff,
            method=method,
            passes=adj_sr > 0.0,
        )
    # Effective track record length in years
    effective_t = (t / 252.0) if (annualized and t >= 30) else float(t)
    effective_t = max(1e-6, effective_t)

    # Standard t-statistic: t_stat = SR * sqrt(T_years)
    t_stat = sr * math.sqrt(effective_t)
    log_tail = log_norm_tail(t_stat)
    # Unadjusted two-sided log p-value
    log_p_unadj = math.log(2.0) + log_tail

    # Multiple testing adjusted log p-value
    if method == "bonferroni":
        log_multiplier = math.log(n_eff)
    elif method == "bhy":
        # Sum_{i=1}^N (1/i) approx ln(N) + gamma
        harmonic = sum(1.0 / i for i in range(1, int(n_trials) + 1))
        log_multiplier = math.log(n_eff) + math.log(harmonic)
    elif method == "holm":
        log_multiplier = math.log(n_eff)
    else:
        raise ValueError(f"Unknown multiple testing method: {method}")

    log_p_adj = log_p_unadj + log_multiplier

    if log_p_adj >= 0.0:  # p_adj >= 1.0
        p_adj = 1.0
        t_haircut = 0.0
        haircut_sr = 0.0
    else:
        p_adj = math.exp(log_p_adj)
        # Invert two-sided adjusted p-value: find t* such that 2 * (1 - Phi(t*)) = p_adj
        # i.e. 1 - Phi(t*) = p_adj / 2 => log(1 - Phi(t*)) = log_p_adj - log(2)
        log_p_half = log_p_adj - math.log(2.0)
        t_haircut = invert_log_p(log_p_half)
        haircut_sr = max(0.0, t_haircut / math.sqrt(effective_t))
    p_unadj = math.exp(log_p_unadj) if log_p_unadj < 0.0 else 1.0
    haircut_pct = max(0.0, min(1.0, 1.0 - (haircut_sr / sr))) if sr > 0.0 else 1.0
    passes = (p_adj <= significance_level) and (haircut_sr > 0.0)

    return HaircutResult(
        haircut_sr=haircut_sr,
        haircut_pct=haircut_pct,
        original_sr=float(sr),
        p_unadj=p_unadj,
        p_adj=p_adj,
        n_trials=int(n_trials),
        n_eff=n_eff,
        method=method,
        passes=passes,
    )


class HaircutSharpeRatio:
    """Class wrapper for Haircut Sharpe Ratio."""

    def __init__(
        self,
        sr: float,
        n_trials: int,
        t: Union[int, float] = 252,
        method: str = "bonferroni",
        correlation: float = 0.0,
        significance_level: float = 0.05,
        annualized: bool = True,
    ) -> None:
        self.result = haircut_sharpe(
            sr=sr,
            n_trials=n_trials,
            t=t,
            method=method,
            correlation=correlation,
            significance_level=significance_level,
            annualized=annualized,
        )

    @classmethod
    def compute(
        sr: float,
        n_trials: int,
        t: Union[int, float] = 252,
        method: str = "bonferroni",
        correlation: float = 0.0,
        significance_level: float = 0.05,
        annualized: bool = True,
    ) -> HaircutResult:
        return haircut_sharpe(
            sr=sr,
            n_trials=n_trials,
            t=t,
            method=method,
            correlation=correlation,
            significance_level=significance_level,
            annualized=annualized,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.result, name)

    def __float__(self) -> float:
        return float(self.result.haircut_sr)

    def __repr__(self) -> str:
        return (
            f"HaircutSharpeRatio(original_sr={self.result.original_sr:.4f}, "
            f"haircut_sr={self.result.haircut_sr:.4f}, "
            f"haircut_pct={self.result.haircut_pct*100:.2f}%, "
            f"passes={self.result.passes})"
        )


# ---------------------------------------------------------------------------
# Comprehensive Anti-Overfitting Audit Suite
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditReport:
    """Comprehensive statistical audit report."""
    dsr_result: DSRResult
    pbo_result: PBOResult
    haircut_result: HaircutResult
    overall_pass: bool
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dsr": self.dsr_result.as_dict(),
            "pbo": self.pbo_result.as_dict(),
            "haircut": self.haircut_result.as_dict(),
            "overall_pass": self.overall_pass,
            "summary": self.summary,
        }


class AntiOverfitAuditor:
    """
    Unified Statistical Anti-Overfitting Audit Engine.
    
    Orchestrates DSR, PBO, and Haircut Sharpe ratio checks over a strategy
    candidate search matrix.
    """

    def __init__(
        self,
        pbo_threshold: float = 0.40,
        min_dsr: float = 0.95,
        significance_level: float = 0.05,
    ) -> None:
        self.pbo_threshold = pbo_threshold
        self.min_dsr = min_dsr
        self.significance_level = significance_level

    def audit(
        self,
        returns_matrix: np.ndarray,
        candidate_idx: Optional[int] = None,
        n_partitions: int = 8,
    ) -> AuditReport:
        """
        Run end-to-end anti-overfitting audit on backtest search matrix.
        
        Parameters:
        -----------
        returns_matrix : np.ndarray
            Shape (T, N) where T = observations, N = strategies tested.
        candidate_idx : int, optional
            Index of the candidate strategy to audit. Defaults to strategy with highest IS Sharpe.
        n_partitions : int, default 8
            Number of partitions for PBO CSCV.
        """
        matrix = np.asarray(returns_matrix, dtype=np.float64)
        T, N = matrix.shape

        # Compute trial Sharpe ratios
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0, ddof=1)
        stds = np.where(stds <= 1e-12, 1e-12, stds)
        srs = means / stds

        if candidate_idx is None:
            candidate_idx = int(np.argmax(srs))

        cand_ret = matrix[:, candidate_idx]
        cand_sr = float(srs[candidate_idx])

        # 1. Deflated Sharpe Ratio
        dsr_res = DeflatedSharpeRatio.compute(
            returns=cand_ret,
            trials=srs,
            significance_level=self.significance_level,
        )

        # 2. Probability of Backtest Overfitting (PBO)
        pbo_res = ProbabilityOfBacktestOverfitting.compute(
            returns_matrix=matrix,
            n_partitions=n_partitions,
            pbo_threshold=self.pbo_threshold,
        )

        # 3. Haircut Sharpe Ratio
        # Estimate average pairwise correlation between strategies
        if N > 1:
            corr_mat = np.corrcoef(matrix, rowvar=False)
            # Average off-diagonal elements
            np.fill_diagonal(corr_mat, np.nan)
            avg_corr = float(np.nanmean(corr_mat))
            avg_corr = max(0.0, avg_corr) if not np.isnan(avg_corr) else 0.0
        else:
            avg_corr = 0.0

        haircut_res = haircut_sharpe(
            sr=cand_sr * math.sqrt(252),  # annualized
            n_trials=N,
            t=T,
            method="bonferroni",
            correlation=avg_corr,
            significance_level=self.significance_level,
            annualized=True,
        )

        overall_pass = dsr_res.passes and pbo_res.passes and haircut_res.passes

        summary = (
            f"Audit Summary: {'PASS' if overall_pass else 'FAIL'} | "
            f"DSR: {dsr_res.dsr:.4f} (p={dsr_res.p_value:.4f}, exp_max={dsr_res.expected_max_sr:.4f}) | "
            f"PBO: {pbo_res.pbo:.4f} (degradation={pbo_res.degradation:.2%}, loss_prob={pbo_res.prob_loss:.2%}) | "
            f"Haircut SR: {haircut_res.haircut_sr:.2f} (original={haircut_res.original_sr:.2f}, penalty={haircut_res.haircut_pct:.1%})"
        )

        return AuditReport(
            dsr_result=dsr_res,
            pbo_result=pbo_res,
            haircut_result=haircut_res,
            overall_pass=overall_pass,
            summary=summary,
        )
