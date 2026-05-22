"""
Probability Foundations for Data Science
=========================================
The distributions, theorems, and simulations that underpin everything else.

Covers:
  1. Common distributions (when to use each)
  2. Central Limit Theorem (demonstrated by simulation)
  3. Law of Large Numbers
  4. Bayes theorem (the foundation of inference)
  5. Monte Carlo methods (estimation by simulation)
  6. Confidence intervals via bootstrap
"""

import numpy as np
from scipy import stats
from typing import Callable, Tuple, Dict


# ── Distribution reference: when to use what ──────────────────────────────────
DISTRIBUTION_GUIDE = {
    "Bernoulli":   "Single yes/no trial (did the user convert?)",
    "Binomial":    "Count of successes in n trials (conversions in 1000 visits)",
    "Poisson":     "Count of rare events in fixed interval (orders per hour)",
    "Normal":      "Continuous, symmetric (heights, measurement errors, CLT result)",
    "Exponential": "Time between events (time until next purchase)",
    "Geometric":   "Trials until first success (visits until first conversion)",
    "Beta":        "Probability of a probability (conversion-rate prior)",
    "Gamma":       "Positive continuous, skewed (wait times, insurance claims)",
    "LogNormal":   "Multiplicative processes (income, revenue, file sizes)",
}


def demonstrate_clt(
    population_dist: str = "exponential",
    sample_sizes: list = None,
    n_simulations: int = 10000,
) -> Dict:
    """
    Central Limit Theorem: the distribution of sample means approaches a
    normal distribution as sample size grows, REGARDLESS of the population
    distribution's shape.

    This is WHY we can use normal-based confidence intervals and t-tests
    even when the underlying data is skewed.
    """
    sample_sizes = sample_sizes or [1, 5, 30, 100]
    np.random.seed(42)

    # Generate from a deliberately non-normal population
    samplers = {
        "exponential": lambda n: np.random.exponential(2, n),
        "uniform":     lambda n: np.random.uniform(0, 10, n),
        "bimodal":     lambda n: np.concatenate([
                            np.random.normal(2, 0.5, n // 2),
                            np.random.normal(8, 0.5, n - n // 2)]),
    }
    sampler = samplers.get(population_dist, samplers["exponential"])

    results = {}
    for n in sample_sizes:
        sample_means = np.array([sampler(n).mean() for _ in range(n_simulations)])
        # Test normality of the sample-mean distribution
        _, p_normal = stats.normaltest(sample_means)
        results[n] = {
            "mean_of_means": round(float(sample_means.mean()), 4),
            "std_of_means":  round(float(sample_means.std()), 4),
            "skewness":      round(float(stats.skew(sample_means)), 4),
            "is_normal":     p_normal > 0.05,
            "normality_p":   round(float(p_normal), 4),
        }

    print(f"CLT demonstration (population: {population_dist})")
    print(f"{'n':>6} {'mean':>10} {'std':>10} {'skew':>8} {'normal?':>9}")
    for n, r in results.items():
        print(f"{n:>6} {r['mean_of_means']:>10.3f} {r['std_of_means']:>10.3f} "
              f"{r['skewness']:>8.3f} {str(r['is_normal']):>9}")
    print("→ As n grows, sample-mean distribution becomes normal (skew → 0)")
    return results


def bayes_theorem(
    prior: float, likelihood: float, false_positive_rate: float
) -> Dict:
    """
    Bayes theorem applied to the classic medical-test problem.

    P(disease | positive test) = P(positive | disease) × P(disease)
                                 / P(positive)

    The counter-intuitive result: even an accurate test produces many false
    positives when the disease is rare. This is the base-rate fallacy.

    Args:
        prior:               P(disease) — base rate / prevalence
        likelihood:          P(positive | disease) — test sensitivity
        false_positive_rate: P(positive | no disease)

    Returns:
        Posterior probability and intermediate terms
    """
    p_disease = prior
    p_no_disease = 1 - prior
    # Total probability of a positive test
    p_positive = (likelihood * p_disease) + (false_positive_rate * p_no_disease)
    posterior = (likelihood * p_disease) / p_positive

    result = {
        "prior_P(disease)":           round(prior, 4),
        "sensitivity_P(+|disease)":   round(likelihood, 4),
        "false_positive_rate":        round(false_positive_rate, 4),
        "P(positive_test)":           round(p_positive, 4),
        "posterior_P(disease|+)":     round(posterior, 4),
    }

    print(f"\nBayes Theorem — Medical Test")
    print(f"  Disease prevalence:        {prior*100:.1f}%")
    print(f"  Test sensitivity:          {likelihood*100:.1f}%")
    print(f"  False positive rate:       {false_positive_rate*100:.1f}%")
    print(f"  → P(disease | positive):   {posterior*100:.1f}%")
    if posterior < 0.5 and prior < 0.05:
        print(f"  ⚠️  Base-rate fallacy: despite an accurate test, most")
        print(f"     positives are FALSE because the disease is rare.")
    return result


def monte_carlo_pi(n_samples: int = 1_000_000) -> float:
    """
    Estimate π by Monte Carlo simulation.
    Throw random darts at a unit square; the fraction landing inside the
    inscribed quarter-circle approximates π/4.

    Demonstrates how simulation solves problems that are hard analytically.
    """
    np.random.seed(42)
    points = np.random.uniform(0, 1, (n_samples, 2))
    inside = (points[:, 0] ** 2 + points[:, 1] ** 2) <= 1
    pi_estimate = 4 * inside.mean()
    print(f"\nMonte Carlo π estimate ({n_samples:,} samples): {pi_estimate:.5f}")
    print(f"  True π: {np.pi:.5f} | Error: {abs(pi_estimate - np.pi):.5f}")
    return pi_estimate


def bootstrap_ci(
    data: np.ndarray, statistic: Callable = np.mean,
    n_bootstrap: int = 10000, ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval — resample with replacement to estimate
    the sampling distribution of any statistic, no distributional assumptions.

    Works for medians, percentiles, correlations — anything, even when
    no closed-form CI exists.
    """
    np.random.seed(42)
    n = len(data)
    boot_stats = np.array([
        statistic(np.random.choice(data, n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    point = float(statistic(data))
    print(f"\nBootstrap {ci*100:.0f}% CI: {point:.4f} [{lower:.4f}, {upper:.4f}]")
    return point, lower, upper


if __name__ == "__main__":
    print("Distribution guide:")
    for dist, use in DISTRIBUTION_GUIDE.items():
        print(f"  {dist:<12} {use}")

    demonstrate_clt("exponential")
    bayes_theorem(prior=0.01, likelihood=0.99, false_positive_rate=0.05)
    monte_carlo_pi(1_000_000)

    np.random.seed(0)
    skewed = np.random.exponential(5, 500)
    bootstrap_ci(skewed, np.median)
