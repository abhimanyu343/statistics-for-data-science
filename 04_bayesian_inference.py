"""
Bayesian Inference
==================
Updating beliefs with evidence — the foundation of modern probabilistic ML.

Covers:
  1. Conjugate priors (Beta-Binomial, Gamma-Poisson, Normal-Normal)
  2. Posterior updating with sequential data
  3. Credible intervals (the Bayesian answer to confidence intervals)
  4. Prior sensitivity analysis
  5. Grid approximation for non-conjugate problems
  6. Comparison: Bayesian vs Frequentist interpretation
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List


class BetaBinomial:
    """
    Beta-Binomial conjugate model — the workhorse for estimating proportions
    (conversion rates, click-through rates, defect rates).

    Prior:      θ ~ Beta(α, β)
    Likelihood: k successes ~ Binomial(n, θ)
    Posterior:  θ | data ~ Beta(α + k, β + n - k)

    The posterior is also Beta — that's what "conjugate" means, and it
    lets us update beliefs in closed form, no simulation needed.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        # Beta(1,1) = uniform prior (no prior knowledge)
        self.alpha = alpha
        self.beta = beta
        self.alpha_0 = alpha   # Remember prior for reporting
        self.beta_0 = beta

    def update(self, successes: int, failures: int) -> "BetaBinomial":
        """Update the posterior with observed data."""
        self.alpha += successes
        self.beta += failures
        return self

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_mode(self) -> float:
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return self.posterior_mean

    @property
    def posterior_variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, ci: float = 0.95) -> Tuple[float, float]:
        """
        Bayesian credible interval. UNLIKE a frequentist CI, this has the
        intuitive interpretation: "there's a 95% probability the true value
        lies in this interval, given the data and prior."
        """
        alpha_tail = (1 - ci) / 2
        lower = stats.beta.ppf(alpha_tail, self.alpha, self.beta)
        upper = stats.beta.ppf(1 - alpha_tail, self.alpha, self.beta)
        return float(lower), float(upper)

    def probability_greater_than(self, threshold: float) -> float:
        """P(θ > threshold) — directly answerable in the Bayesian framework."""
        return float(1 - stats.beta.cdf(threshold, self.alpha, self.beta))

    def summary(self) -> Dict:
        ci = self.credible_interval()
        return {
            "prior":            f"Beta({self.alpha_0}, {self.beta_0})",
            "posterior":        f"Beta({self.alpha:.0f}, {self.beta:.0f})",
            "posterior_mean":   round(self.posterior_mean, 4),
            "posterior_mode":   round(self.posterior_mode, 4),
            "credible_95":      (round(ci[0], 4), round(ci[1], 4)),
        }


class GammaPoisson:
    """
    Gamma-Poisson conjugate model — for estimating rates (events per unit time).
    Use cases: orders per hour, support tickets per day, defects per batch.

    Prior:      λ ~ Gamma(α, β)
    Likelihood: counts ~ Poisson(λ)
    Posterior:  λ | data ~ Gamma(α + Σx, β + n)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def update(self, counts: List[int]) -> "GammaPoisson":
        self.alpha += sum(counts)
        self.beta += len(counts)
        return self

    @property
    def posterior_mean(self) -> float:
        return self.alpha / self.beta

    def credible_interval(self, ci: float = 0.95) -> Tuple[float, float]:
        alpha_tail = (1 - ci) / 2
        lower = stats.gamma.ppf(alpha_tail, self.alpha, scale=1/self.beta)
        upper = stats.gamma.ppf(1 - alpha_tail, self.alpha, scale=1/self.beta)
        return float(lower), float(upper)


def grid_approximation(
    data_successes: int, data_trials: int,
    prior_fn=None, grid_points: int = 1000
) -> Dict:
    """
    Grid approximation — compute the posterior numerically when no conjugate
    prior exists. Discretise the parameter space, compute prior × likelihood
    at each point, normalise.

    This is the conceptual basis for MCMC: when grids don't scale to many
    dimensions, we sample instead.
    """
    grid = np.linspace(0, 1, grid_points)
    # Prior (default uniform)
    prior = prior_fn(grid) if prior_fn else np.ones(grid_points)
    # Likelihood: Binomial PMF at each candidate θ
    likelihood = stats.binom.pmf(data_successes, data_trials, grid)
    # Posterior ∝ prior × likelihood
    unnormalised = prior * likelihood
    posterior = unnormalised / (unnormalised.sum())

    # Posterior summaries
    post_mean = float(np.sum(grid * posterior))
    cumsum = np.cumsum(posterior)
    ci_lower = float(grid[np.searchsorted(cumsum, 0.025)])
    ci_upper = float(grid[np.searchsorted(cumsum, 0.975)])

    return {
        "posterior_mean": round(post_mean, 4),
        "credible_95":    (round(ci_lower, 4), round(ci_upper, 4)),
        "map_estimate":   round(float(grid[posterior.argmax()]), 4),
    }


if __name__ == "__main__":
    print("1. Beta-Binomial: estimating a conversion rate")
    print("-" * 50)
    # Start with weak prior, observe 18 conversions in 200 visits
    model = BetaBinomial(alpha=1, beta=1)
    model.update(successes=18, failures=182)
    for k, v in model.summary().items():
        print(f"  {k:<18}: {v}")
    print(f"  P(conversion > 5%): {model.probability_greater_than(0.05)*100:.1f}%")

    print("\n2. Sequential updating (beliefs evolve with data)")
    print("-" * 50)
    seq_model = BetaBinomial(alpha=1, beta=1)
    batches = [(2, 18), (5, 45), (12, 88)]  # (successes, failures) per week
    for week, (s, f) in enumerate(batches, 1):
        seq_model.update(s, f)
        ci = seq_model.credible_interval()
        print(f"  Week {week}: mean={seq_model.posterior_mean:.3f}, "
              f"95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\n3. Gamma-Poisson: estimating orders per hour")
    print("-" * 50)
    rate_model = GammaPoisson(alpha=1, beta=1)
    rate_model.update([12, 15, 9, 18, 14, 11])  # orders in 6 hours
    ci = rate_model.credible_interval()
    print(f"  Estimated rate: {rate_model.posterior_mean:.2f} orders/hour")
    print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\n4. Grid approximation (matches conjugate result)")
    print("-" * 50)
    grid_result = grid_approximation(18, 200)
    for k, v in grid_result.items():
        print(f"  {k:<16}: {v}")
