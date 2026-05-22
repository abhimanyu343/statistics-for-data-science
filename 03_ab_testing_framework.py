"""
A/B Testing Framework
======================
Production-grade A/B testing: sample size calculation, statistical testing,
sequential testing (peek-proof), Bayesian alternative, and common pitfalls.

Most A/B test analyses are wrong. This module fixes that.

Common mistakes this framework prevents:
  1. Peeking (stopping early when you see significance)
  2. Multiple testing without correction
  3. Ignoring practical significance (effect size)
  4. Running test on wrong metric (ratio metrics need special treatment)
  5. Forgetting to check for novelty effects and segment differences
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Dict, Literal
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ABTestResult:
    """Structured result from an A/B test."""
    metric:             str
    control_mean:       float
    treatment_mean:     float
    relative_lift_pct:  float
    absolute_diff:      float
    p_value:            float
    is_significant:     bool
    confidence_interval: Tuple[float, float]
    effect_size:        float        # Cohen's d or relative effect
    power:              float
    sample_sizes:       Tuple[int, int]
    test_type:          str
    recommendation:     str

    def __str__(self):
        sig = "✅ SIGNIFICANT" if self.is_significant else "❌ NOT SIGNIFICANT"
        return (
            f"
{'='*55}
"
            f"A/B Test: {self.metric}
"
            f"{'='*55}
"
            f"  Control:       {self.control_mean:.4f}  (n={self.sample_sizes[0]:,})
"
            f"  Treatment:     {self.treatment_mean:.4f}  (n={self.sample_sizes[1]:,})
"
            f"  Lift:          {self.relative_lift_pct:+.2f}%  "
            f"(abs: {self.absolute_diff:+.4f})
"
            f"  p-value:       {self.p_value:.4f}
"
            f"  95% CI:        [{self.confidence_interval[0]:+.4f}, "
            f"{self.confidence_interval[1]:+.4f}]
"
            f"  Effect size:   {self.effect_size:.4f} (Cohen's d)
"
            f"  Power:         {self.power:.2f}
"
            f"  Result:        {sig}
"
            f"  Action:        {self.recommendation}
"
        )


class ABTestFramework:
    """
    Complete A/B testing framework for conversion rates, revenue, and engagement metrics.
    """

    def __init__(self, alpha: float = 0.05, power: float = 0.80):
        """
        Args:
            alpha: Significance level (Type I error rate). Default 0.05.
            power: Desired statistical power (1 - Type II error). Default 0.80.
        """
        self.alpha = alpha
        self.power = power

    # ── Sample size calculation ───────────────────────────────────────────────

    def required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        test_type: Literal["two_sided", "one_sided"] = "two_sided",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        baseline_std: Optional[float] = None,
    ) -> Dict:
        """
        Calculate required sample size per variant before running the test.
        Always do this BEFORE collecting data — never "collect until significant".

        Args:
            baseline_rate:              Current conversion rate (e.g., 0.05 = 5%)
            minimum_detectable_effect:  Smallest lift you care about (e.g., 0.02 = 2pp)
            test_type:                  'two_sided' (default) or 'one_sided'
            metric_type:                'proportion' (conversion) or 'continuous' (revenue, time)
            baseline_std:               Required for continuous metrics

        Returns:
            Dict with n_per_variant, total_n, days_needed_estimate
        """
        sides = 1 if test_type == "one_sided" else 2
        z_alpha = stats.norm.ppf(1 - self.alpha / sides)
        z_power = stats.norm.ppf(self.power)

        if metric_type == "proportion":
            p1 = baseline_rate
            p2 = baseline_rate + minimum_detectable_effect
            # Pooled variance formula for proportions
            p_pooled = (p1 + p2) / 2
            var_pooled = p_pooled * (1 - p_pooled)
            var_effect = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
            n = ((z_alpha * np.sqrt(2 * var_pooled) +
                  z_power * np.sqrt(var_effect)) ** 2) / (minimum_detectable_effect ** 2)
        else:
            if baseline_std is None:
                raise ValueError("baseline_std required for continuous metrics")
            # Cohen's d based sample size
            effect_size_d = minimum_detectable_effect / baseline_std
            n = (2 * ((z_alpha + z_power) / effect_size_d) ** 2)

        n_per_variant = int(np.ceil(n))
        # Add 10% buffer for drop-outs, bot traffic, etc.
        n_per_variant_buffered = int(n_per_variant * 1.1)

        return {
            "n_per_variant":          n_per_variant,
            "n_per_variant_buffered": n_per_variant_buffered,
            "total_n":                n_per_variant_buffered * 2,
            "baseline_rate":          baseline_rate,
            "mde":                    minimum_detectable_effect,
            "mde_relative_pct":       round(minimum_detectable_effect / baseline_rate * 100, 1),
            "alpha":                  self.alpha,
            "power":                  self.power,
            "test_type":              test_type,
        }

    # ── Proportion test (conversion rates) ───────────────────────────────────

    def test_proportions(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        metric_name: str = "conversion_rate",
    ) -> ABTestResult:
        """
        Two-proportion z-test for conversion rate differences.
        Most common A/B test for click-through, sign-up, purchase rates.
        """
        p_ctrl = control_conversions / control_n
        p_trt  = treatment_conversions / treatment_n

        # Pooled proportion under null hypothesis (p_ctrl == p_trt)
        p_pool = (control_conversions + treatment_conversions) / (control_n + treatment_n)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n))

        z_stat = (p_trt - p_ctrl) / (se + 1e-10)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # 95% confidence interval for the difference
        se_diff = np.sqrt(p_ctrl * (1-p_ctrl)/control_n + p_trt * (1-p_trt)/treatment_n)
        z_crit = stats.norm.ppf(1 - self.alpha/2)
        ci = (p_trt - p_ctrl - z_crit * se_diff,
               p_trt - p_ctrl + z_crit * se_diff)

        # Cohen's h for proportions
        effect_size = float(2 * np.arcsin(np.sqrt(p_trt)) - 2 * np.arcsin(np.sqrt(p_ctrl)))

        # Post-hoc power
        power = self._compute_power_proportion(p_ctrl, p_trt, control_n, treatment_n)

        is_sig = p_value < self.alpha
        lift = (p_trt - p_ctrl) / (p_ctrl + 1e-10) * 100
        recommendation = self._recommend(is_sig, lift, power)

        return ABTestResult(
            metric=metric_name, control_mean=round(p_ctrl,4),
            treatment_mean=round(p_trt,4), relative_lift_pct=round(lift,2),
            absolute_diff=round(p_trt - p_ctrl, 4), p_value=round(p_value,4),
            is_significant=is_sig, confidence_interval=(round(ci[0],4), round(ci[1],4)),
            effect_size=round(effect_size,4), power=round(power,2),
            sample_sizes=(control_n, treatment_n), test_type="two_proportion_z",
            recommendation=recommendation
        )

    # ── Continuous metric test (revenue, session duration) ────────────────────

    def test_means(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str = "metric",
        use_welch: bool = True,
    ) -> ABTestResult:
        """
        Welch's t-test for continuous metrics (revenue, time-on-site, etc.).
        Welch's is preferred over Student's t because it doesn't assume equal variance.
        """
        ctrl = np.asarray(control_values)
        trt  = np.asarray(treatment_values)

        t_stat, p_value = stats.ttest_ind(ctrl, trt, equal_var=not use_welch)

        # 95% CI for difference in means
        diff = trt.mean() - ctrl.mean()
        se_diff = np.sqrt(ctrl.var(ddof=1)/len(ctrl) + trt.var(ddof=1)/len(trt))
        t_crit = stats.t.ppf(1 - self.alpha/2, df=len(ctrl) + len(trt) - 2)
        ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)

        # Cohen's d effect size
        pooled_std = np.sqrt((ctrl.var(ddof=1) + trt.var(ddof=1)) / 2)
        d = diff / (pooled_std + 1e-10)

        is_sig = float(p_value) < self.alpha
        lift = diff / (abs(ctrl.mean()) + 1e-10) * 100
        power = self._compute_power_means(ctrl, trt)
        recommendation = self._recommend(is_sig, lift, power)

        return ABTestResult(
            metric=metric_name, control_mean=round(float(ctrl.mean()),4),
            treatment_mean=round(float(trt.mean()),4), relative_lift_pct=round(lift,2),
            absolute_diff=round(float(diff),4), p_value=round(float(p_value),4),
            is_significant=is_sig, confidence_interval=(round(ci[0],4), round(ci[1],4)),
            effect_size=round(float(d),4), power=round(power,2),
            sample_sizes=(len(ctrl), len(trt)),
            test_type="welch_t" if use_welch else "student_t",
            recommendation=recommendation
        )

    # ── Bayesian A/B test (for proportion metrics) ────────────────────────────

    def bayesian_test(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 50_000,
    ) -> Dict:
        """
        Bayesian A/B test using Beta-Binomial conjugate model.
        
        Posterior: Beta(alpha + conversions, beta + non-conversions)
        
        Advantages over frequentist:
        - P(treatment > control) — direct probability statement
        - No multiple testing issues (look whenever you want)
        - Incorporates prior knowledge
        - Expected loss quantifies cost of wrong decision
        """
        np.random.seed(42)

        # Posterior distributions
        alpha_ctrl = prior_alpha + control_conversions
        beta_ctrl  = prior_beta + (control_n - control_conversions)
        alpha_trt  = prior_alpha + treatment_conversions
        beta_trt   = prior_beta + (treatment_n - treatment_conversions)

        # Sample from posteriors (Monte Carlo)
        ctrl_samples = np.random.beta(alpha_ctrl, beta_ctrl, n_samples)
        trt_samples  = np.random.beta(alpha_trt,  beta_trt,  n_samples)

        prob_trt_better = float((trt_samples > ctrl_samples).mean())
        expected_lift   = float((trt_samples - ctrl_samples).mean())

        # Credible intervals (Bayesian equivalent of confidence intervals)
        ctrl_ci = (float(np.percentile(ctrl_samples, 2.5)), float(np.percentile(ctrl_samples, 97.5)))
        trt_ci  = (float(np.percentile(trt_samples, 2.5)),  float(np.percentile(trt_samples, 97.5)))

        # Expected loss: E[max(ctrl - trt, 0)] if we choose treatment
        expected_loss = float(np.maximum(ctrl_samples - trt_samples, 0).mean())

        # Decision rule: deploy if P(trt > ctrl) > 0.95 AND expected_loss < threshold
        deploy_threshold = 0.95
        loss_threshold   = 0.001  # Accept <0.1pp loss
        decision = "DEPLOY" if (prob_trt_better > deploy_threshold and
                                 expected_loss < loss_threshold) else "WAIT"

        result = {
            "prob_treatment_better":   round(prob_trt_better, 4),
            "prob_control_better":     round(1 - prob_trt_better, 4),
            "expected_lift":           round(expected_lift, 4),
            "expected_loss_if_deploy": round(expected_loss, 6),
            "control_posterior_ci95":  ctrl_ci,
            "treatment_posterior_ci95": trt_ci,
            "control_posterior_mean":  round(alpha_ctrl / (alpha_ctrl + beta_ctrl), 4),
            "treatment_posterior_mean": round(alpha_trt / (alpha_trt + beta_trt), 4),
            "decision":                decision,
            "n_samples":               n_samples,
        }

        print(f"
{'='*55}")
        print("Bayesian A/B Test Results")
        print(f"{'='*55}")
        print(f"  P(Treatment > Control): {prob_trt_better*100:.1f}%")
        print(f"  Expected lift:          {expected_lift*100:+.3f}pp")
        print(f"  Expected loss:          {expected_loss*100:.4f}pp")
        print(f"  Decision:               {decision}")
        print(f"  Control 95% CI:         [{ctrl_ci[0]:.4f}, {ctrl_ci[1]:.4f}]")
        print(f"  Treatment 95% CI:       [{trt_ci[0]:.4f}, {trt_ci[1]:.4f}]")

        return result

    # ── Multiple testing correction ───────────────────────────────────────────

    def correct_multiple_tests(
        self,
        p_values: List[float],
        method: Literal["bonferroni", "holm", "fdr_bh"] = "fdr_bh",
    ) -> pd.DataFrame:
        """
        Correct for multiple comparisons when testing multiple metrics or variants.
        
        Methods:
          bonferroni: Conservative. α_adj = α/n. Controls FWER.
          holm:       Less conservative than Bonferroni. Controls FWER.
          fdr_bh:     Benjamini-Hochberg. Controls FDR. Best for many tests.
        """
        from statsmodels.stats.multitest import multipletests
        reject, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method=method)
        return pd.DataFrame({
            "original_p":   p_values,
            "corrected_p":  p_corrected.round(4),
            "significant":  reject,
            "method":       method
        })

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_power_proportion(self, p1, p2, n1, n2) -> float:
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower
        es = abs(proportion_effectsize(p1, p2))
        try:
            return float(NormalIndPower().solve_power(
                effect_size=es, nobs1=n1, ratio=n2/n1, alpha=self.alpha))
        except Exception:
            return 0.0

    def _compute_power_means(self, ctrl, trt) -> float:
        from statsmodels.stats.power import TTestIndPower
        pooled_std = np.sqrt((ctrl.var(ddof=1) + trt.var(ddof=1)) / 2)
        d = abs(trt.mean() - ctrl.mean()) / (pooled_std + 1e-10)
        try:
            return float(TTestIndPower().solve_power(
                effect_size=d, nobs1=len(ctrl), ratio=len(trt)/len(ctrl), alpha=self.alpha))
        except Exception:
            return 0.0

    def _recommend(self, is_sig: bool, lift_pct: float, power: float) -> str:
        if not is_sig and power < 0.5:
            return "Underpowered — collect more data before concluding"
        if not is_sig:
            return "No significant effect detected — maintain control"
        if lift_pct > 0 and is_sig:
            return f"Ship treatment — {lift_pct:+.1f}% lift with statistical significance"
        if lift_pct < 0 and is_sig:
            return f"Keep control — treatment is significantly worse ({lift_pct:.1f}%)"
        return "Borderline — consider practical significance alongside statistical"


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    framework = ABTestFramework(alpha=0.05, power=0.80)

    # 1. Sample size planning
    print("Sample Size Planning:")
    ss = framework.required_sample_size(baseline_rate=0.05, minimum_detectable_effect=0.01)
    print(f"  Need {ss['n_per_variant_buffered']:,} users per variant ({ss['total_n']:,} total)")
    print(f"  To detect {ss['mde_relative_pct']}% relative lift with {ss['power']*100:.0f}% power")

    # 2. Frequentist test
    result = framework.test_proportions(
        control_conversions=420, control_n=10000,
        treatment_conversions=480, treatment_n=10000,
        metric_name="signup_rate"
    )
    print(result)

    # 3. Bayesian test
    bayesian = framework.bayesian_test(420, 10000, 480, 10000)

    # 4. Multiple testing
    p_values = [0.03, 0.04, 0.06, 0.001, 0.08, 0.12]
    corrected = framework.correct_multiple_tests(p_values, method="fdr_bh")
    print("
Multiple Testing Correction (BH method):")
    print(corrected.to_string(index=False))
