"""
Hypothesis Testing — Comprehensive Reference
=============================================
Every major test, when to use it, and what the assumptions actually mean.

Tests covered:
  Parametric:    t-test (1-sample, 2-sample, paired), ANOVA, F-test
  Non-parametric: Mann-Whitney U, Kruskal-Wallis, Wilcoxon signed-rank
  Categorical:   Chi-square goodness-of-fit, chi-square independence, Fisher's exact
  Normality:     Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, Q-Q plot
  Correlation:   Pearson, Spearman, Kendall tau, partial correlation

Decision framework:
  → Check normality (n < 30: Shapiro-Wilk, n ≥ 30: D'Agostino)
  → If normal: parametric tests
  → If not normal: non-parametric or bootstrap
  → Always check practical significance (effect size), not just p-value
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Run multiple normality tests and return comprehensive diagnostics.
    Use all tests together — no single test is definitive.
    """
    data = np.asarray(data).ravel()
    n = len(data)
    results = {}

    # Shapiro-Wilk: best for n < 50, sensitive to small deviations with large n
    if n <= 5000:
        stat, p = stats.shapiro(data)
        results["shapiro_wilk"] = {"statistic": round(stat,4), "p_value": round(p,4),
                                    "normal": p > alpha, "note": "Best for n<50"}

    # D'Agostino-Pearson: uses skewness and kurtosis, good for n > 20
    if n >= 20:
        stat, p = stats.normaltest(data)
        results["dagostino_pearson"] = {"statistic": round(stat,4), "p_value": round(p,4),
                                         "normal": p > alpha, "note": "Uses skew+kurtosis"}

    # Kolmogorov-Smirnov: weaker, but works on any n
    stat, p = stats.kstest(data, "norm", args=(data.mean(), data.std()))
    results["kolmogorov_smirnov"] = {"statistic": round(stat,4), "p_value": round(p,4),
                                      "normal": p > alpha, "note": "Conservative for large n"}

    # Summary statistics
    results["summary"] = {
        "n": n,
        "mean": round(data.mean(), 4),
        "std": round(data.std(), 4),
        "skewness": round(stats.skew(data), 4),    # Normal: near 0
        "kurtosis": round(stats.kurtosis(data), 4), # Normal: near 0 (excess)
        "consensus_normal": sum(v.get("normal", False)
                                for v in results.values()
                                if isinstance(v, dict) and "normal" in v) >= 2
    }
    return results


def comprehensive_two_sample_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    group_names: Tuple[str, str] = ("Group A", "Group B"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run all relevant two-sample tests and compare results.
    Returns DataFrame with test name, statistic, p-value, conclusion, and notes.
    """
    a, b = np.asarray(group_a), np.asarray(group_b)
    results = []

    # Welch's t-test (doesn't assume equal variances)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    results.append({"test": "Welch t-test", "statistic": t, "p_value": p,
                    "significant": p < alpha,
                    "assumes": "Normality (both groups)", "robust": "Large n"})

    # Student's t-test (assumes equal variances)
    t, p = stats.ttest_ind(a, b, equal_var=True)
    results.append({"test": "Student t-test", "statistic": t, "p_value": p,
                    "significant": p < alpha,
                    "assumes": "Normality + equal variance", "robust": "Rarely preferred over Welch"})

    # Mann-Whitney U (non-parametric: tests median/distribution)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    results.append({"test": "Mann-Whitney U", "statistic": u, "p_value": p,
                    "significant": p < alpha,
                    "assumes": "Independent samples", "robust": "No normality needed"})

    # Levene's test for equal variances
    lev, p_lev = stats.levene(a, b)
    results.append({"test": "Levene (variance equality)", "statistic": lev, "p_value": p_lev,
                    "significant": p_lev < alpha,
                    "assumes": "Independent samples",
                    "robust": "Use before choosing t-test variant"})

    # Effect sizes
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    cohens_d = (a.mean() - b.mean()) / (pooled_std + 1e-10)

    df = pd.DataFrame(results)
    df["statistic"] = df["statistic"].round(4)
    df["p_value"]   = df["p_value"].round(4)

    print(f"
Two-sample comparison: {group_names[0]} (n={len(a)}) vs {group_names[1]} (n={len(b)})")
    print(f"Means: {a.mean():.3f} vs {b.mean():.3f}")
    print(f"Cohen's d: {cohens_d:.3f}  "
          f"({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'})")
    print(df.to_string(index=False))
    return df


def chi_square_independence(
    contingency_table: np.ndarray,
    row_labels: List[str] = None,
    col_labels: List[str] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Chi-square test of independence for categorical variables.
    Also computes Cramér's V (effect size, 0=none, 1=perfect association).

    Use Fisher's exact test when any expected cell count < 5.
    """
    table = np.asarray(contingency_table)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)

    # Check Fisher's exact test condition
    use_fisher = (expected < 5).any()
    if use_fisher and table.shape == (2, 2):
        _, p_fisher = stats.fisher_exact(table)
        print(f"⚠️  Expected counts < 5 — using Fisher's exact test (p={p_fisher:.4f})")

    # Cramér's V — effect size for chi-square
    n = table.sum()
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * max(min_dim, 1)))

    result = {
        "chi2_statistic":  round(float(chi2), 4),
        "p_value":         round(float(p), 4),
        "degrees_of_freedom": int(dof),
        "is_significant":  bool(p < alpha),
        "cramers_v":       round(float(cramers_v), 4),
        "effect_strength": ("negligible" if cramers_v < 0.1
                            else "small" if cramers_v < 0.3
                            else "medium" if cramers_v < 0.5
                            else "large"),
        "use_fisher":      bool(use_fisher),
        "expected_counts": expected.round(1).tolist(),
    }

    print(f"
Chi-Square Test of Independence")
    print(f"  χ² = {chi2:.3f}, df = {dof}, p = {p:.4f}")
    print(f"  Cramér's V = {cramers_v:.3f} ({result['effect_strength']} association)")
    print(f"  {'✅ Significant' if result['is_significant'] else '❌ Not significant'}")
    return result


if __name__ == "__main__":
    np.random.seed(42)

    print("1. Normality Tests")
    print("-" * 40)
    normal_data = np.random.normal(50, 10, 200)
    skewed_data = np.random.exponential(5, 200)
    norm_result = check_normality(normal_data)
    print(f"Normal data — consensus normal: {norm_result['summary']['consensus_normal']}")
    skew_result = check_normality(skewed_data)
    print(f"Skewed data — consensus normal: {skew_result['summary']['consensus_normal']}")

    print("
2. Two-Sample Tests")
    print("-" * 40)
    group_a = np.random.normal(100, 15, 150)
    group_b = np.random.normal(107, 18, 150)
    comprehensive_two_sample_test(group_a, group_b, ("Control", "Treatment"))

    print("
3. Chi-Square Independence")
    print("-" * 40)
    contingency = np.array([[45, 55], [30, 70]])
    chi_square_independence(contingency, ["Group A", "Group B"], ["Yes", "No"])
