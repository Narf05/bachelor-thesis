# Analysis Plan

## Purpose

This document defines the analysis that will later be implemented in a Jupyter notebook in `code/analysis/`.
The notebook should not only produce figures and tables, but also explain clearly what each step means, why it is done, and how the resulting outputs should be interpreted.

The central goal of the analysis is to study vocabulary richness estimation across many texts or speakers, with particular emphasis on estimator behavior, sample size effects, and the role of the low-frequency spectrum.

Cross-domain comparison is not the main focus, but it is still worth including as a secondary perspective because it may reveal meaningful structural differences between corpora.

## General Principles For The Notebook

The notebook should be written as an explanation, not just as a sequence of calculations.

Each major section should contain:

- a short introduction explaining the question being addressed
- a description of the data being used in that section
- an explanation of the metric or estimator being plotted
- an interpretation paragraph after every figure or table
- a short transition explaining why the next section follows logically

Every plot or table must answer a concrete question. No figure should appear without a short explanation of:

- what is shown
- why it matters
- what the main pattern is
- what the limitations are

The notebook should distinguish carefully between:

- observed vocabulary quantities
- estimated unseen vocabulary quantities
- total estimated vocabulary richness
- effects driven by sample size
- effects driven by the frequency spectrum

## Proposed Notebook Structure

## 1. Introduction And Scope

This opening section should explain:

- the overall problem of estimating total vocabulary richness from incomplete samples
- why observed vocabulary alone is not enough
- why different estimators may behave differently
- why the present analysis focuses on empirical behavior rather than only formal definitions

This section should also state clearly that:

- the main aim is to compare estimator behavior on real corpora
- domain comparison is secondary
- real corpora do not provide true total vocabulary, so strict error metrics are not available without a proxy or simulation framework

## 2. Data Overview

This section should summarize the data that will be analyzed.

It should describe:

- which corpora are included
- what a single observational unit is
- the filtering rules, especially the minimum-word threshold when active
- how many texts or speakers remain after filtering
- the range of token counts across the retained data

Useful outputs in this section:

- a summary table of corpus sizes and number of retained texts
- a histogram or density plot of token counts
- a histogram or density plot of observed vocabulary size
- possibly a table of minimum, median, mean, and maximum token counts by corpus

Explanation required:

- why token count distribution matters for estimator analysis
- why very small texts are problematic
- how the retained sample defines the rest of the study

## 3. Observed Lexical Diversity Before Estimation

Before introducing richness estimators, the notebook should show what is directly observed in the data.

This section should focus on:

- observed vocabulary size
- token count
- type-token style summaries if useful
- sample coverage measures that are already available

Useful plots:

- observed vocabulary size versus token count
- type-token ratio or another normalized observed diversity indicator versus token count
- coverage versus token count

Purpose:

- establish the baseline empirical structure
- show how much variation is already visible before extrapolation
- prepare the reader to understand why estimators may diverge

Explanation required:

- observed vocabulary usually grows with sample size
- normalized measures can still be unstable at low counts
- coverage indicates how incomplete a sample may be

## 4. Main Estimator Comparison: Estimated Richness Versus Sample Size

This should be the core section of the notebook.

The main question is:

How do the different richness estimators behave as sample size changes?

The key plot should be:

- token count on the x-axis
- estimated total vocabulary richness on the y-axis
- one curve or visual layer per estimator

This should be shown in two forms:

- raw point cloud or scatter view to display full variation
- smoothed or binned average curves to show the general trend

Why this is useful:

- it reveals how strongly estimates depend on sample size
- it shows whether some estimators grow faster or more slowly than others
- it helps identify unstable estimators at low or medium token counts

Explanation required:

- whether estimator differences shrink or widen as token count increases
- whether any estimator appears systematically conservative or aggressive
- whether some estimators are highly sensitive to sparse frequency spectra

## 5. Observed Versus Estimated Vocabulary

This section should compare what is observed directly with what each model infers beyond the sample.

Useful plots:

- observed vocabulary size versus estimated total richness
- observed vocabulary size versus estimated unseen vocabulary
- ratio of estimated total richness to observed vocabulary

Purpose:

- quantify the extrapolation gap
- show how far each estimator moves beyond the observed data
- identify methods that produce especially large or small adjustments

Explanation required:

- whether estimators remain close to observed richness for high-coverage texts
- whether extrapolation is larger when the sample is more incomplete
- whether the extrapolation factor differs systematically by estimator

## 6. Frequency Spectrum Analysis

This section should study the connection between low-frequency counts and richness estimation.

This is one of the most important interpretive parts of the notebook because the estimators are driven heavily by rare-word behavior.

The section should include at least the following analyses:

- `f_1` versus estimated total richness
- `f_1` versus estimated unseen vocabulary
- `f_1` and `f_2` jointly, if feasible in a clear visual form
- a summary of `f_1` to `f_5`, either individually or as grouped rare-frequency mass

Possible visualizations:

- scatter plots for `f_1` versus estimate
- scatter plots for normalized singleton intensity, such as `f_1 / n` or `f_1 / S_obs`
- a plot using the sum of `f_1` through `f_5`
- a heatmap or faceted view if multiple rare-frequency predictors are shown together

Purpose:

- explain what drives high richness estimates
- distinguish true estimator behavior from simple sample-size scaling
- connect abstract formulas to interpretable corpus statistics

Explanation required:

- why singleton counts matter
- why raw `f_1` may be confounded with sample size
- why normalized rare-word summaries may be more informative
- how different estimators react to similar frequency-spectrum inputs

## 7. Coverage-Based Interpretation

Because sample coverage is stored and conceptually important, it should have its own dedicated section.

Useful plots:

- coverage versus estimated total richness
- coverage versus estimated unseen vocabulary
- coverage versus ratio of estimated richness to observed richness

Purpose:

- connect the extrapolation logic to how complete the sample appears to be
- show whether richer estimates are associated with lower observed coverage
- provide a more principled interpretation of why some texts are difficult to estimate

Explanation required:

- what coverage means in practical terms
- why low coverage implies more uncertainty about unseen vocabulary
- whether estimator disagreement is greatest at low coverage

## 8. Estimator Agreement And Disagreement

This section should compare estimators directly with one another.

Useful outputs:

- pairwise scatter plots of estimator A versus estimator B
- difference plots between estimators
- ratio plots between estimators
- a summary table of correlation or agreement across estimators

Purpose:

- determine whether estimators mostly tell the same story
- identify the situations in which they diverge most
- separate robust conclusions from method-dependent conclusions

Explanation required:

- which estimators tend to agree strongly
- where disagreement becomes substantial
- whether disagreement is associated with token count, low coverage, or high singleton burden

## 9. Secondary Domain Comparison

This section is optional in emphasis but recommended because it may reveal interesting differences in data structure.

The point is not to make domain comparison the main contribution, but to show whether corpus type changes the behavior of observed or estimated richness.

Useful outputs:

- boxplots or violin plots of token count by corpus
- boxplots or violin plots of observed vocabulary size by corpus
- boxplots or violin plots of estimated richness by corpus
- rare-frequency summaries by corpus
- coverage distributions by corpus

Purpose:

- contextualize the main findings
- show whether corpora differ in ways that could influence estimator performance
- help interpret whether a given estimator is responding to domain-specific lexical structure

Explanation required:

- that domain differences are descriptive rather than the central question
- that some apparent richness differences may be driven by sample-size distribution
- that strong conclusions about domain effects should be cautious unless sample-size effects are controlled

## 10. Real-Data Summary Tables

Because true error is not available on the real corpora, this section should not present classical accuracy measures such as RMSE or MAE.

Instead, it should summarize estimator behavior descriptively.

Useful table contents:

- estimator name
- mean estimated richness
- median estimated richness
- standard deviation or interquartile range
- mean ratio of estimated richness to observed richness
- fraction of cases with very large extrapolation
- fraction of undefined or unstable outputs if such cases occur

Purpose:

- provide a concise summary of estimator behavior on real data
- complement the visualizations with numerical comparisons
- document stability, spread, and extrapolation intensity

Explanation required:

- why these are descriptive summaries rather than true error statistics
- what large spread implies
- which estimators appear more stable or more aggressive on real data

## 11. Evaluation Strategy For True Error

This section should not yet compute code-based results unless a later notebook is specifically dedicated to evaluation, but the analytical logic should be stated clearly.

The notebook should explain that true error metrics require a reference truth or pseudo-truth.

Possible evaluation designs to describe:

- simulation from known distributions
- subsampling from very large texts and treating the full sample as a reference
- incremental prediction experiments where early-sample estimates are compared to later observed growth

This section should explain that a table like RMSE, MAE, bias, or interval coverage only becomes defensible in one of those settings.

Purpose:

- make the methodological limitation explicit
- prevent overclaiming from real data
- prepare a later extension of the analysis if needed

## 12. Final Discussion

The notebook should end with a synthesis of the main findings.

It should discuss:

- which estimator patterns are robust
- how strongly sample size shapes the results
- what the rare-frequency spectrum reveals
- whether coverage helps explain estimator disagreement
- what the practical lessons are for vocabulary richness estimation on empirical corpora

It should also state the main limitations:

- lack of true ground truth on real corpora
- dependence on preprocessing and filtering choices
- possible domain heterogeneity
- sensitivity of some methods to sparse or noisy frequency spectra

## Standards For Figure And Table Explanations

Every figure should be followed by a written interpretation block containing:

- the main empirical pattern
- the likely statistical explanation
- the methodological implication
- at least one limitation or caution

Every table should be followed by a short explanation of:

- what the columns represent
- which entries deserve attention
- how the table supports or qualifies the narrative

No figure should be left unexplained, and no statistical output should appear without a sentence describing how it contributes to the larger research question.

## Recommended Narrative Flow

The later notebook should read like a coherent argument:

1. Describe the data and its sample-size structure.
2. Show what is directly observed.
3. Introduce the richness estimates.
4. Compare estimators across sample size.
5. Explain estimator behavior through the frequency spectrum and coverage.
6. Add secondary domain context.
7. Summarize estimator behavior in descriptive tables.
8. Clarify why true error requires a separate evaluation design.
9. Conclude with the practical lessons.

## Final Deliverable Expectation

The eventual notebook should be readable as a self-contained analysis document.

It should not function only as a technical script. It should explain:

- what is being analyzed
- why each step is needed
- how each figure should be interpreted
- what conclusions are justified
- what remains uncertain

This is especially important because the value of the analysis will come not only from producing estimator outputs, but from interpreting them carefully and transparently.
