## MODIFIED Requirements

### Requirement: Correlation Confidence Interval Calculation

The system SHALL provide a function `calculate_correlation_ci(r, n, confidence_level=0.95)` in `cross_experiment_analysis.py` that computes confidence intervals for correlation coefficients using the Fisher z-transformation method:

1. **Validate inputs**:
   - Raise `ValueError` if r is not in [-1, 1] (and not NaN)
   - Raise `ValueError` if confidence_level is not in (0, 1) exclusive range
2. **Transform to z-scale**: z = arctanh(r) = 0.5 × ln((1+r)/(1-r))
3. **Compute standard error**: SE_z = 1 / √(n-3)
4. **Compute z-scale CI**: z ± z_{α/2} × SE_z where α = 1 - confidence_level
5. **Back-transform to r-scale**: r = tanh(z)
6. **Clamp bounds**: Ensure -1 ≤ ci_low ≤ ci_high ≤ 1

Edge case handling:
- **r = ±1.0**: Return (r, r) as point mass (arctanh undefined at boundaries)
- **n < 4**: Return (NaN, NaN) as variance undefined (n-3 in denominator)
- **r = NaN**: Return (NaN, NaN) without raising validation error

Documentation:
- Docstring SHALL note that for Spearman correlations, Fisher z-based CI is accurate for n >= 10; for 4 <= n < 10, results are approximate
- Docstring SHALL cross-reference `calculate_correlation_confidence_intervals` for DataFrame-based operations

#### Scenario: CI for moderate correlation with adequate sample size

- **WHEN** r = 0.5 and n = 20 with confidence_level = 0.95
- **THEN** CI bounds are approximately (0.06, 0.78)
- **AND** interval is symmetric on z-scale but asymmetric on r-scale

#### Scenario: CI for zero correlation

- **WHEN** r = 0.0 and n = 30 with confidence_level = 0.95
- **THEN** CI bounds are approximately (-0.36, 0.36)
- **AND** interval contains zero (expected for null correlation)

#### Scenario: CI for perfect correlation

- **WHEN** r = 1.0 and n = 50
- **THEN** CI is (1.0, 1.0) as point mass
- **AND** no mathematical error from arctanh(1.0) = infinity

#### Scenario: CI for negative perfect correlation

- **WHEN** r = -1.0 and n = 50
- **THEN** CI is (-1.0, -1.0) as point mass
- **AND** no mathematical error from arctanh(-1.0) = -infinity

#### Scenario: CI undefined for very small n

- **WHEN** r = 0.5 and n = 3
- **THEN** CI is (NaN, NaN)
- **AND** warning is logged about insufficient sample size for CI

#### Scenario: CI for near-boundary correlation

- **WHEN** r = 0.99 and n = 100
- **THEN** CI bounds are valid: ci_low < 0.99 < ci_high <= 1.0
- **AND** bounds are clamped to [-1, 1] if numerical precision causes overshoot

#### Scenario: Higher confidence level widens interval

- **WHEN** same r and n but confidence_level changes from 0.95 to 0.99
- **THEN** CI width increases by factor of approximately z_{0.005}/z_{0.025} ≈ 2.576/1.96 ≈ 1.31

#### Scenario: Invalid correlation coefficient raises error

- **WHEN** r = 1.5 (outside valid range)
- **THEN** function raises ValueError with message indicating r must be in [-1, 1]
- **AND** error occurs before any computation

#### Scenario: Invalid confidence level raises error

- **WHEN** confidence_level = 0 or confidence_level = 1.0 or confidence_level = -0.5
- **THEN** function raises ValueError with message indicating confidence_level must be in (0, 1)
- **AND** error occurs before any computation

#### Scenario: NaN correlation does not raise validation error

- **WHEN** r = NaN
- **THEN** function returns (NaN, NaN) without raising ValueError
- **AND** this allows graceful handling of missing correlation data
