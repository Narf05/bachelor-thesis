# Blueprint For Thesis Sections 6, 7, And 8

This document describes the missing empirical part of the thesis: Results, Discussion, and Conclusion. It is based on the current thesis draft in `thesis/bachelor_thesis_new.pdf` / `thesis/main.tex`, where Sections 6, 7, and 8 are still placeholders, and on the theoretical material already written in Sections 2-5.

The goal is to turn the previous sections into a clear empirical story:

1. replicate the Shakespeare vocabulary-richness analysis as a baseline,
2. summarize the available corpora by language,
3. estimate vocabulary richness for each language and corpus,
4. show rarefaction and extrapolation behavior,
5. compare languages and estimators in tables and graphs,
6. discuss what can and cannot be concluded from real corpora without ground-truth vocabulary sizes.

## Main Thesis Logic

The earlier sections already define the statistical language needed for the results:

- Each word token is an individual.
- Each distinct word type is a species.
- The observed vocabulary is `S_obs`.
- The unobserved vocabulary is `f_0`.
- The total vocabulary richness is `S = S_obs + f_0`.
- The frequency counts `f_1, f_2, ...` drive the estimators.
- Sample coverage explains how complete a sample is.
- Rarefaction and extrapolation show how richness changes with sampling effort.
- Classical estimators give simple lower-bound or correction-based estimates.
- Breakaway uses frequency-ratio regression.
- k-monotone and completely monotone models use shape constraints on the frequency distribution.

The empirical sections should not repeat all theory. They should use this notation to answer:

> Given finite speech or text samples, how much vocabulary richness do different estimators infer, and how stable are those estimates across languages, corpora, and sample sizes?

## Section 6: Results

Section 6 should be the largest new section. It should be structured as a sequence of empirical questions, not as a list of methods.

### 6.1 Shakespeare Replication Baseline

Purpose:

Use Shakespeare as a known reference case before moving to the multilingual speech/dialogue corpora. This verifies that the pipeline and estimators produce plausible values on a dataset already used in the literature.

Important scope rule:

Shakespeare is not part of the cross-language spoken-language comparison. It is a written literary corpus and should only be used as a replication and validation baseline. When comparing English, French, and German spoken language, exclude Shakespeare and use only the spoken/dialogue corpora.

Required table:

| Dataset | Tokens `n` | Observed types `S_obs` | `f_1` | `f_2` | Coverage | Estimator | Estimated unseen `f_0_hat` | Estimated total `S_hat` | Ratio `S_hat / S_obs` |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|

Rows should include at least:

- Chao1
- iChao1
- ACE
- Jackknife 1
- Jackknife 2
- Breakaway, if stable
- k-monotone estimator, if implemented
- completely monotone estimator, if implemented

Required graph:

- Bar chart or dot plot of Shakespeare `S_hat` by estimator.
- Include `S_obs` as a horizontal reference line.
- If published estimates are available from Balabdaoui and Kulagina or Chee and Wang, include them as comparison markers.

Interpretation to write:

- Whether the estimators are close to the published Shakespeare values.
- Which estimators are conservative and which extrapolate strongly.
- Whether the singleton/doubleton counts explain the amount of extrapolation.
- Why Shakespeare is useful as a baseline but not identical to spoken data.

### 6.2 Corpus And Language Overview

Purpose:

Before comparing models, describe the data. This is essential because estimator behavior depends strongly on sample size and on the number of rare words.

Language grouping:

| Language | Corpora |
|---|---|
| English | IMSDb, BNC, Santa Barbara Corpus |
| French | CLAPI |
| German | DGD |

Shakespeare should be reported separately from this table or marked explicitly as a written replication baseline, not as part of the English spoken-language data.

Required language-level table:

| Language | Corpora included | Number of speakers/texts | Total tokens | Total observed types | Mean tokens per speaker | Median tokens per speaker | Mean observed types | Mean words per speaker/text | Mean coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Required corpus-level table:

| Corpus | Language | Unit of analysis | Number of units | Total tokens | Median tokens | Min tokens | Max tokens | Median `S_obs` | Median `f_1 / n` | Median coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Recommended graphs:

- Histogram or density plot of token counts by language.
- Boxplot of token counts by corpus.
- Boxplot of observed vocabulary `S_obs` by corpus.
- Coverage distribution by language.

Interpretation to write:

- Whether languages/corpora are comparable in sample size.
- Whether some corpora have many tiny speakers/texts.
- Whether coverage differs by language.
- Why raw vocabulary comparisons are misleading without standardization.

### 6.3 Per-Language Estimator Results

Purpose:

For each language, show what each model predicts. This should be the main bridge from theory to empirical claims.

Required table for each language:

| Language | Estimator | Mean `S_hat` | Median `S_hat` | IQR `S_hat` | Mean `f_0_hat` | Median `S_hat / S_obs` | Failed/undefined fits |
|---|---|---:|---:|---:|---:|---:|---:|

Recommended graph for each language:

- Boxplot or violin plot of `S_hat` by estimator.
- Scatter plot of token count `n` versus `S_hat`, colored by estimator.
- Scatter plot of coverage versus `S_hat / S_obs`, colored by estimator.

Interpretation to write:

- Which estimators give similar results.
- Which estimators diverge.
- Whether divergence happens mainly for low coverage or small samples.
- Whether estimates are dominated by corpus composition.

### 6.4 Completely Monotone Choice And Justification

Purpose:

The thesis argues that completely monotone estimation is justified when the selected degree of monotonicity becomes large. The results must show this choice explicitly, not simply state that CM was used.

If finite-k model selection is implemented, include:

| Language | Corpus | Unit | Selected `k` | AIC/BIC/log-likelihood value | CM estimate | Finite-k estimate | Difference |
|---|---|---|---:|---:|---:|---:|---:|

Recommended graphs:

- Line plot of AIC or BIC versus `k` for representative examples.
- Histogram of selected `k` values by language or corpus.
- Scatter plot comparing finite-k estimates against completely monotone estimates.

If finite-k model selection is not implemented, write this transparently:

- Classical estimators and Breakaway can be analyzed now.
- CM results require an additional implementation step.
- The thesis can still motivate CM theoretically, but empirical CM claims should not be made without the fitted model outputs.

Interpretation to write:

- If selected `k` often reaches the largest tested value, this supports using the completely monotone limit.
- If selected `k` is often small, CM should be presented as one estimator among others, not as the obvious final model.
- If CM is numerically unstable for small samples, restrict it to samples above a token threshold.

### 6.5 Rarefaction And Extrapolation Curves

Purpose:

Show how observed richness grows with sample size and how much additional vocabulary is expected with more sampling.

Required outputs:

- Rarefaction/extrapolation curve for Shakespeare.
- Representative curve for each language.
- Combined language-level curves after standardization.

Recommended graph design:

- x-axis: sample size or sampling effort.
- y-axis: expected richness.
- vertical line: observed sample size.
- left side: rarefaction.
- right side: extrapolation.
- confidence bands if implemented.

For language comparison, include two versions:

- sample-size-based curves,
- coverage-based curves.

Interpretation to write:

- Whether curves are still rising steeply at the observed sample size.
- Which languages/corpora appear more incomplete.
- Whether coverage-based comparison changes the ranking.
- Why extrapolation should be interpreted cautiously far beyond the observed sample size.

### 6.6 Cross-Language Comparison

Purpose:

Compare languages together while accounting for sample size, corpus type, and estimator choice.

Scope:

Use only spoken or dialogue-based corpora in this section. Exclude Shakespeare because the research question is about spoken language and because Shakespeare is a written literary corpus. Shakespeare belongs in Section 6.1 as a baseline for method replication, not in the English cross-language group.

Required table:

| Language | Estimator | Median `S_obs` | Median `S_hat` | Median `f_0_hat` | Median coverage | Median `S_hat / S_obs` |
|---|---|---:|---:|---:|---:|---:|

Recommended graphs:

- Grouped bar chart of median `S_hat` by language and estimator.
- Heatmap with languages as rows, estimators as columns, and median `S_hat / S_obs` as values.
- Boxplot of `S_hat / S_obs` by language and estimator.
- Coverage-standardized richness comparison across languages.

Interpretation to write:

- Whether apparent language differences remain after standardization.
- Whether English dominates because it has more spoken/dialogue corpora and larger samples.
- Whether French and German results are limited by corpus availability.
- Whether estimator rankings are stable across languages.
- That Shakespeare was deliberately excluded from this comparison to keep the comparison focused on spoken language.

### 6.7 Estimator Comparison Across All Data

Purpose:

Compare the models themselves, independently of one language.

Required table:

| Estimator | Mean `S_hat / S_obs` | Median `S_hat / S_obs` | IQR | Failure rate | Typical behavior |
|---|---:|---:|---:|---:|---|

Recommended graphs:

- Pairwise scatter plots of estimator estimates.
- Difference plot: estimator minus Chao1.
- Ratio plot: estimator divided by Chao1.
- Heatmap of estimator correlations.

Interpretation to write:

- Which estimators agree enough to support robust conclusions.
- Which estimators are aggressive.
- Which estimators are conservative lower bounds.
- Which estimators fail or become unstable.

## Section 7: Discussion

The discussion should not introduce many new results. It should explain what the results mean.

Recommended structure:

1. **Shakespeare replication**
   - Did the estimates match the literature?
   - What does this say about implementation quality?

2. **Vocabulary richness differs from observed vocabulary**
   - `S_obs` is always an underestimate.
   - The gap depends on singleton burden and coverage.

3. **Estimator behavior**
   - Chao1 and iChao1 are lower-bound style methods.
   - ACE adjusts for rare-species heterogeneity.
   - Jackknife methods are simple but may overshoot.
   - Breakaway can use more of the frequency spectrum but can be unstable.
   - CM/k-monotone estimators are theoretically attractive if the shape constraint is empirically supported.

4. **Language comparison**
   - Discuss differences only after mentioning sample-size and corpus-composition limitations.
   - Avoid claiming that one language has a larger true vocabulary unless standardized results clearly support it.

5. **Rarefaction and extrapolation**
   - Steep curves imply many unseen types remain.
   - Coverage-based comparison is fairer when token counts differ.
   - Far extrapolation is uncertain.

6. **Limitations**
   - No true ground truth for full speaker vocabulary.
   - Different corpora have different genres and collection methods.
   - Tokenization choices affect type counts.
   - No lemmatization means inflected forms count separately.
   - Very small speakers/texts may produce unstable estimates.
   - CM results require careful numerical validation.

## Section 8: Conclusion

The conclusion should answer the research question directly and briefly.

Suggested conclusion points:

- The thesis studied active vocabulary richness as a species-richness problem.
- Shakespeare served as a replication baseline and was not included in the spoken-language cross-language comparison.
- Across real speech/dialogue corpora, observed vocabulary was incomplete, as shown by singleton counts and sample coverage.
- Estimators differed substantially, especially for low-coverage samples.
- Rarefaction and extrapolation curves showed that many corpora were still far from saturation.
- Cross-language comparisons are possible, but must be interpreted through sample-size and coverage standardization.
- The most defensible claims are about estimator behavior and sample incompleteness, not exact true vocabulary sizes.

## Implementation Checklist

Minimum viable results:

- Shakespeare estimator table.
- Corpus/language overview table.
- Per-language estimator summary tables.
- Per-language estimator boxplots.
- Rarefaction/extrapolation examples.
- Cross-language estimator comparison table and graph.

Strong thesis results:

- Published Shakespeare comparison.
- Coverage-based standardization.
- Estimator agreement/disagreement analysis.
- CM/k-monotone selection plots.
- Failure/instability reporting for every estimator.

Do not claim unless implemented:

- CM is best.
- One language has objectively larger vocabulary richness.
- Estimated totals are true vocabulary sizes.
- Real-data error rates such as RMSE/MAE, unless a simulation or pseudo-ground-truth experiment is added.

## Recommended File Outputs

Suggested outputs in `results/`:

- `shakespeare_estimator_table.csv`
- `language_corpus_overview.csv`
- `per_language_estimator_summary.csv`
- `cross_language_estimator_summary.csv`
- `estimator_failure_summary.csv`

Suggested figures:

- `shakespeare_estimator_comparison.png`
- `tokens_by_language.png`
- `coverage_by_language.png`
- `estimator_boxplots_by_language.png`
- `rarefaction_extrapolation_by_language.png`
- `coverage_standardized_language_comparison.png`
- `estimator_correlation_heatmap.png`
- `cm_k_selection_examples.png`, if CM/k-monotone model selection is implemented.

Output organization:

The main figures should be saved in `results/` as image files, preferably `.png`. It is fine, and probably clearer, to use different files for different tasks instead of forcing all outputs into one notebook or one combined result file. For example, Shakespeare replication, language overview, estimator comparison, rarefaction/extrapolation, and cross-language comparison can each have their own CSV tables and PNG figures.

## My Recommendation

The proposed plan is strong and coherent. The best narrative is:

1. verify the methods on Shakespeare,
2. describe the multilingual spoken/dialogue data honestly,
3. compare estimators within each language,
4. show rarefaction/extrapolation to make incompleteness visible,
5. compare languages only after standardizing or at least controlling for sample size and coverage,
6. discuss estimator disagreement as a central result rather than a problem to hide.

The main caution is that cross-language comparison is the riskiest part. For this comparison, Shakespeare should be excluded because it is written literature rather than spoken language. English still has several spoken/dialogue corpora and much more variety, while French and German appear to have fewer source families. Therefore, the thesis should frame language comparisons as descriptive and coverage-aware, not as definitive statements about the inherent vocabulary richness of languages.
