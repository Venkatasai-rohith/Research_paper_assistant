# RAG Evaluation Report --- Latest Recorded Run

**Run ID:** `20261011_174235`\
**Generated at:** 2026-10-11 17:47:09\
**Evaluation scope:** Planner and hybrid retrieval

## Summary metrics

  ------------------------------------------------------------------------
  Evaluation       Metric                           Score Details
  category                                                
  ---------------- ---------------- --------------------- ----------------
  Planner          Intent accuracy                 100.0% 25/25 correct

  Planner          Paper-routing                   100.0% 25/25 correct
                   accuracy                               

  Hybrid retrieval Mean Precision@5                 0.736 22 retrieval
                                                          test cases

  Hybrid retrieval Mean Hit@5                       1.000 22 retrieval
                                                          test cases

  Hybrid retrieval Mean Reciprocal                  0.955 22 retrieval
                   Rank (MRR)                             test cases
  ------------------------------------------------------------------------

## Metric definitions and limitations

-   **Precision@5:** Average fraction of the five retrieved chunks that
    are labeled relevant.
-   **Hit@5:** Whether at least one relevant chunk appears among the top
    five results, averaged across queries.
-   **MRR:** Average reciprocal rank of the first relevant result.

Hit@5 is not the same metric as Recall@5. Recall@5 measures the fraction
of all ground-truth relevant items retrieved among the top five and
requires the corresponding relevance judgments. This run reports Hit@5.

Retrieval relevance labels in the current setup are keyword-based and
have not yet been established as human-verified semantic relevance
judgments. Interpret the results with that limitation in mind.

## Scope

This run contains planner and retrieval results only. It does not
contain QA-answer LLM-judge results or summary LLM-judge results. Their
absence here does not mean those categories scored zero or failed.

For the broader suite description and the separately retained historical
QA/summary samples, see `backend/eval/EVALUATION_REPORT.md`.

## Reproduction

Run from `backend/`:

``` powershell
python -m eval.run_eval --retrieval --save
```
