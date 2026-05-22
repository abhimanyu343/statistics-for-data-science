"""
Causal Inference
================
Correlation isn't causation — these methods estimate what actually CAUSES what.
Essential for product decisions: "did our feature cause the lift, or was it seasonality?"

Covers:
  1. Potential outcomes framework (treatment effects)
  2. Randomised experiments (the gold standard)
  3. Difference-in-Differences (DiD) — natural experiments
  4. Propensity score matching — observational data
  5. Regression discontinuity — threshold-based assignment
  6. Simpson's paradox (why aggregates mislead)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


def average_treatment_effect(
    treated_outcomes: np.ndarray, control_outcomes: np.ndarray
) -> Dict:
    """
    Estimate the Average Treatment Effect (ATE) from a randomised experiment.

    ATE = E[Y | treated] - E[Y | control]

    Randomisation is what makes this causal: because assignment is random,
    treatment and control groups are comparable on all confounders
    (observed AND unobserved), so the difference in outcomes is the
    causal effect.
    """
    ate = float(treated_outcomes.mean() - control_outcomes.mean())
    # Standard error and t-test
    se = np.sqrt(treated_outcomes.var(ddof=1) / len(treated_outcomes) +
                 control_outcomes.var(ddof=1) / len(control_outcomes))
    t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes, equal_var=False)
    ci = (ate - 1.96 * se, ate + 1.96 * se)

    return {
        "ate":              round(ate, 4),
        "relative_lift_pct": round(100 * ate / control_outcomes.mean(), 2),
        "std_error":        round(float(se), 4),
        "p_value":          round(float(p_value), 4),
        "ci_95":            (round(ci[0], 4), round(ci[1], 4)),
        "significant":      bool(p_value < 0.05),
    }


def difference_in_differences(
    df: pd.DataFrame, outcome: str = "outcome",
    group: str = "treated", time: str = "post",
) -> Dict:
    """
    Difference-in-Differences (DiD) — estimate causal effect when you have
    a treatment group and control group observed before AND after treatment.

    DiD = (treated_after - treated_before) - (control_after - control_before)

    The key assumption is PARALLEL TRENDS: absent treatment, both groups
    would have followed the same trend. We subtract out the control group's
    change to remove time effects (seasonality, market shifts).

    Use case: a feature rolled out to some users but not others, measure
    its effect controlling for general time trends.
    """
    means = df.groupby([group, time])[outcome].mean()

    treated_before  = means.loc[(1, 0)]
    treated_after   = means.loc[(1, 1)]
    control_before  = means.loc[(0, 0)]
    control_after   = means.loc[(0, 1)]

    treated_diff = treated_after - treated_before
    control_diff = control_after - control_before
    did_estimate = treated_diff - control_diff

    # Estimate via regression for standard errors (interaction term = DiD)
    df = df.copy()
    df["interaction"] = df[group] * df[time]
    import statsmodels.formula.api as smf
    model = smf.ols(f"{outcome} ~ {group} + {time} + interaction", data=df).fit()

    return {
        "did_estimate":       round(float(did_estimate), 4),
        "treated_change":     round(float(treated_diff), 4),
        "control_change":     round(float(control_diff), 4),
        "regression_coef":    round(float(model.params["interaction"]), 4),
        "p_value":            round(float(model.pvalues["interaction"]), 4),
        "significant":        bool(model.pvalues["interaction"] < 0.05),
    }


def propensity_score_matching(
    df: pd.DataFrame, treatment: str, outcome: str, covariates: list,
) -> Dict:
    """
    Propensity Score Matching — estimate causal effects from OBSERVATIONAL
    data (no randomisation) by matching treated units to similar control units.

    Propensity score = P(treated | covariates), estimated via logistic regression.
    Matching on this single score balances all the covariates, approximating
    a randomised experiment.

    Caveat: only controls for OBSERVED confounders. Unobserved confounders
    can still bias the estimate (unlike randomisation).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[covariates])
    t = df[treatment].values

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000).fit(X, t)
    df = df.copy()
    df["propensity"] = ps_model.predict_proba(X)[:, 1]

    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    # Nearest-neighbour matching on propensity score
    matched_effects = []
    control_ps = control["propensity"].values
    for _, row in treated.iterrows():
        # Find closest control unit by propensity score
        idx = np.argmin(np.abs(control_ps - row["propensity"]))
        matched_control_outcome = control.iloc[idx][outcome]
        matched_effects.append(row[outcome] - matched_control_outcome)

    att = float(np.mean(matched_effects))  # Average Treatment effect on Treated
    se = float(np.std(matched_effects) / np.sqrt(len(matched_effects)))

    return {
        "att":              round(att, 4),
        "std_error":        round(se, 4),
        "n_treated":        len(treated),
        "n_control":        len(control),
        "ci_95":            (round(att - 1.96*se, 4), round(att + 1.96*se, 4)),
    }


def simpsons_paradox_demo() -> None:
    """
    Simpson's Paradox: a trend that appears in groups can REVERSE when
    the groups are combined. Why you must think about confounders.

    Classic example: a treatment looks worse overall but is better in
    every subgroup, because it was given more often to sicker patients.
    """
    # Treatment A vs B, by severity
    data = pd.DataFrame({
        "treatment": ["A","A","B","B"],
        "severity":  ["mild","severe","mild","severe"],
        "recovered": [81, 192, 234, 55],
        "total":     [87, 263, 270, 80],
    })
    data["recovery_rate"] = data["recovered"] / data["total"]

    print("By subgroup:")
    print(data[["treatment","severity","recovery_rate"]].to_string(index=False))

    # Aggregate (ignoring severity)
    agg = data.groupby("treatment").apply(
        lambda g: g["recovered"].sum() / g["total"].sum(), include_groups=False)
    print(f"\nAggregate recovery rates:")
    print(f"  Treatment A: {agg['A']:.3f}")
    print(f"  Treatment B: {agg['B']:.3f}")
    print(f"\n→ A wins in BOTH subgroups, but B wins overall!")
    print(f"  The confounder (severity) was unevenly distributed.")


if __name__ == "__main__":
    np.random.seed(42)

    print("1. Randomised Experiment (ATE)")
    print("-" * 50)
    control = np.random.normal(100, 20, 1000)
    treated = np.random.normal(108, 20, 1000)
    result = average_treatment_effect(treated, control)
    for k, v in result.items():
        print(f"  {k:<18}: {v}")

    print("\n2. Difference-in-Differences")
    print("-" * 50)
    n = 500
    did_df = pd.DataFrame({
        "treated": np.repeat([0, 1], n * 2),
        "post":    np.tile(np.repeat([0, 1], n), 2),
    })
    # Treated group gets +5 effect post-treatment, both have +3 time trend
    did_df["outcome"] = (50 + 3 * did_df["post"]
                         + 5 * did_df["treated"] * did_df["post"]
                         + np.random.normal(0, 5, len(did_df)))
    did_result = difference_in_differences(did_df)
    for k, v in did_result.items():
        print(f"  {k:<18}: {v}")

    print("\n3. Simpson's Paradox")
    print("-" * 50)
    simpsons_paradox_demo()
