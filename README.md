# Temporal Data Fusion: Correcting Selection Bias with Lagged Population Data

## Overview

This project investigates a temporal data fusion problem in which recent individual-level observations come from a biased sample, while more representative population data are only available from earlier time periods.

The objective is to estimate current target-population quantities by leveraging historical information about the sampling mechanism to correct selection bias.

This repository contains my independently developed mathematical derivations, literature review, and Python simulations exploring binary and continuous outcome settings, density-ratio estimation, and Hájek-style weighting.

## Research Problem

Suppose two data sources are available:

- **Biased source (S_t):** Individual-level observations available through the current time T, potentially affected by nonrandom sampling.
- **Target population (P_t):** More representative population observations available only through an earlier time T-k.

The goal is to estimate the current population mean,

$$
\mu_T = E_{P_T}[Y]
$$

without directly observing the target-population distribution at time T.

The central challenge is determining whether historical information about selection bias can be used to correct recent observations when the population distribution changes over time.

## Research Contributions and Methodology

### 1. Binary and Continuous Outcome Derivations

Derived the relationship between observed sample proportions and target-population proportions for binary outcomes using Bayes' rule:

$$
p_t = \frac{s_t}{\alpha+s_t-\alpha s_t}
$$

where alpha represents the ratio of response probabilities for positive and negative outcomes.

Extended the framework to continuous outcomes by expressing the target-population mean as a normalized inverse-probability-weighted estimator:

$$
\hat{\mu}_T =
\frac{\sum_i w(Y_{i,T})Y_{i,T}}
{\sum_i w(Y_{i,T})}
$$

Under a stationary relative sampling-bias assumption, the weights can be derived from historical density ratios. I also formulated the estimator through weighted loss minimization and examined its connection to the Hájek estimator.

### 2. Density-Ratio Estimation

Investigated direct density-ratio estimation using probabilistic classification rather than separately estimating the two probability densities.

Implemented logistic regression with polynomial features to distinguish historical biased-source observations from target-population observations. The predicted class probabilities are converted into weights for estimating current population quantities.

### 3. Synthetic Simulation

Developed Python simulations under the following data-generating process:

- Historical population: P_0 ~ N(0, 1)
- Current population: P_1 ~ N(0.5, 1)
- Response probability: alpha(y) = 1 / (1 + exp(-y))
- True current population mean: 0.5

Compared an oracle estimator using the analytically derived density ratio against a classifier-based estimator using estimated weights.

Evaluated estimation accuracy, distribution alignment after reweighting, sensitivity to sample size, and variability across repeated simulations.

### 4. Literature Review and Extensions

Reviewed related work on Hájek estimation, density-ratio estimation, and time-series methods to investigate the statistical assumptions underlying the framework and potential extensions to time-varying sampling bias.

## Repository Contents

- **data_fusion (5).pdf:** Research notes containing the problem formulation, mathematical derivations, estimator properties, literature review, and simulation setup.
- **data_fusion (3).ipynb:** Python implementation of synthetic data generation, density-ratio estimation, weighted estimators, and diagnostic visualizations.

## Implementation

**Language:** Python

**Libraries:** NumPy, SciPy, scikit-learn, Matplotlib

The Jupyter notebook generates synthetic data and does not require an external dataset.

## Research Scope

This repository documents exploratory methodological research rather than a finalized estimation framework.

The theoretical development and simulations rely on assumptions about the stability of the relative sampling mechanism across time. Potential extensions include incorporating covariates, modeling time-varying selection bias, and establishing statistical guarantees under broader conditions.
