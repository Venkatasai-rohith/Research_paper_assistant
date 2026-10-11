# RAG Evaluation Suite --- Benchmark Overview

**Project:** Multi-Paper Research Assistant\
**Last updated:** 2026-10-11\
**Purpose:** Overview of the configured evaluation suite and the results
currently retained in `backend/eval/results/`.

> **Interpretation note:** The suite contains 87 configured test cases,
> but the retained artifacts do not establish that all 87 cases were
> executed successfully in one complete run. Results below are separated
> by evaluation run and by the amount of recorded evidence available.

## 1. Evaluation suite composition

The test suite defines 87 cases across four categories:

  Category       Configured cases What it tests
  ------------ ------------------ --------------------------------------------------
  QA                           22 Questions grounded in individual research papers
  Comparison                   20 Cross-paper comparison and synthesis
  Summary                      20 Paper-summary quality
  Planner                      25 Intent classification and paper routing
  **Total**                **87** Configured test-case inventory

A configured case count is not the same as a successfully completed
execution count. Use the run-specific evidence below to interpret
results.

## 2. Latest recorded planner and retrieval run

**Artifact:** `eval_20261011_174235.json`\
**Readable report:** `eval_20261011_174235_report.md`\
**Generated:** 2026-10-11 17:47:09

  ------------------------------------------------------------------------
  Component        Metric                          Result Recorded scope
  ---------------- ---------------- --------------------- ----------------
  Planner          Intent accuracy         100.0% (25/25) 25 planner cases

  Planner          Paper-routing           100.0% (25/25) 25 planner cases
                   accuracy                               

  Hybrid retrieval Mean Precision@5                 0.736 22 retrieval
                                                          cases

  Hybrid retrieval Mean Hit@5                       1.000 22 retrieval
                                                          cases

  Hybrid retrieval Mean Reciprocal                  0.955 22 retrieval
                   Rank (MRR)                             cases
  ------------------------------------------------------------------------

### Metric interpretation

-   **Precision@5** measures the proportion of the top-five retrieved
    chunks judged relevant, averaged across retrieval test cases.
-   **Hit@5** indicates whether at least one relevant chunk was found
    among the top five results for a query, averaged across queries.
-   **MRR** measures how highly the first relevant result is ranked.

**Hit@5 and Recall@5 are not interchangeable.** Recall@5 requires a
ground-truth count of all relevant items for each query and measures the
fraction retrieved in the top five. The retained latest report records
Hit@5, so this overview does not relabel it as Recall@5.

Retrieval relevance labels were keyword-based in the current evaluation
setup and have not been established as human-verified semantic relevance
judgments. Interpret retrieval scores accordingly.

## 3. Previously recorded QA-answer quality results

**Artifact:** `eval_20261009_192255.json`\
**CSV:** `eval_20261009_192255_qa_answers.csv`

The retained QA-answer artifact contains three judged answer records.
The historical report summarized them as follows:

  Metric                              Historical reported result Scope
  --------------------------------- ---------------------------- --------------------
  Answer faithfulness (LLM judge)                       4.67 / 5 3 recorded answers
  Answer relevance (LLM judge)                          5.00 / 5 3 recorded answers
  Average citations per answer                              2.67 3 recorded answers

These are results from a small recorded sample, not evidence that all 22
QA-answer cases completed. LLM-judge scores are evaluator judgments, not
independently verified ground truth.

## 4. Previously recorded summary-quality results

**Artifacts:** `eval_20261009_192500.json` and
`eval_20261009_192924.json`\
**CSV exports:** `eval_20261009_192500_summaries.csv` and
`eval_20261009_192924_summaries.csv`

The retained artifacts represent three recorded summary evaluations in
total. The earlier overview reported these historical aggregate scores:

  Metric                         Historical reported result Scope
  ---------------------------- ---------------------------- ---------------------------
  Topic coverage (LLM judge)                       5.00 / 5 Historical summary sample
  Text coherence (LLM judge)                       5.00 / 5 Historical summary sample
  Conciseness (LLM judge)                          4.00 / 5 Historical summary sample

Treat these as historical sample results. They do not establish that all
20 configured summary cases completed. Re-run the summary evaluation
successfully before presenting them as a current full-suite benchmark.

## 5. Comparison evaluation status

The suite defines 20 comparison cases. The retained report artifacts
summarized here do not establish a complete, successfully finished
comparison evaluation. Do not mark the comparison pipeline as passing
until its run output and saved results have been checked.

## 6. Reproducibility

Run commands from the `backend/` directory.

``` powershell
# Planner and retrieval
python -m eval.run_eval --retrieval --save

# QA-answer evaluation (LLM judge)
python -m eval.run_eval --answer --save

# Summary-quality evaluation (LLM judge)
python -m eval.run_eval --summary --save
```

A previous summary invocation stopped with:

``` text
TypeError: eval_summaries() takes 2 positional arguments but 3 were given
```

Therefore, that invocation is a failed run and must not be counted as a
successful summary evaluation. Fix the function-call/signature mismatch,
rerun the command, and verify that new JSON/CSV/report artifacts were
written before updating the summary results here.

The commands above evaluate selected categories; do not assume a single
full 87-case execution unless the runner explicitly runs all categories
and the saved artifact confirms their results.

## 7. Artifact guide

  ---------------------------------------------------------------------------
  Artifact                                Purpose
  --------------------------------------- -----------------------------------
  `EVALUATION_REPORT.md`                  This overview; separates configured
                                          suite size from recorded results

  `eval_20261011_174235.json`             Raw latest planner/retrieval
                                          results

  `eval_20261011_174235_report.md`        Human-readable report for the
                                          latest planner/retrieval run

  `eval_20261009_192255.json`             Historical QA-answer judge records

  `eval_20261009_192255_qa_answers.csv`   QA-answer records in tabular form

  `eval_20261009_192500.json` and         Historical summary judge records
  `eval_20261009_192924.json`             

  `eval_20261009_192500_summaries.csv`    Summary records in tabular form
  and                                     
  `eval_20261009_192924_summaries.csv`    
  ---------------------------------------------------------------------------

## 8. Current conclusion

The latest retained run shows perfect planner intent and routing
accuracy across 25 cases, along with strong recorded retrieval metrics
across 22 cases. The retained QA-answer and summary results are smaller
historical samples. A complete end-to-end benchmark claim should wait
until the relevant evaluation categories have been run successfully and
their outputs verified.
