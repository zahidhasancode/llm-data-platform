# Problem Statement

This document describes the operational and engineering challenges that motivate an internal LLM data and training platform. It focuses on problems only; solutions are out of scope here.

## LLM data pipelines are hard to manage

Data for LLM training and evaluation comes from many sources: raw logs, human annotations, synthetic data, third-party datasets, and ad hoc exports. Pipelines that ingest, clean, and format these inputs are often one-off scripts, spread across repos and runbooks, with no shared schema or validation. Changing one step can break downstream consumers in non-obvious ways. There is no single place to see what data exists, how it was produced, or who owns it. As a result, teams spend significant time debugging pipelines and reconciling inconsistent data formats instead of improving models.

## Dataset versioning is missing

Teams frequently refer to "the dataset" or "the latest export" without a clear notion of version. The same logical dataset may exist as multiple files (e.g. different sampling seeds, date ranges, or preprocessing choices) with similar names. It is difficult to know exactly which rows and columns were used for a given training run or evaluation. Without versioning, reproducing a past result or rolling back to a known-good dataset is unreliable. Auditing what data was used for a deployed model is also difficult.

## Reproducible training runs are difficult

Training runs depend on many factors: dataset version, code version, hyperparameters, hardware, and framework versions. These are often specified in scattered config files, env vars, or command-line arguments. Re-running "the same" training to verify a result or to resume from a checkpoint often fails because some dependency has changed. There is no standard way to capture the full provenance of a run (data + code + config + environment) so that runs can be reproduced or compared.

## Tracking model quality over time is difficult

Evaluations are run at different times, on different slices of data, and with different metrics or prompts. Results are stored in notebooks, spreadsheets, or one-off dashboards. Comparing a new model to a baseline or to last week’s model requires manual reconciliation of evaluation setups. It is hard to answer questions like: "Did this change improve quality on production-like traffic?" or "Which model was best on task X as of date Y?" without a consistent evaluation pipeline and a single place to record and compare results.

## Need for a unified data and training platform

Internal ML teams need to iterate quickly on data and models while keeping experiments reproducible and quality measurable. Today, that work is fragmented across ad hoc pipelines, unversioned datasets, non-reproducible runs, and inconsistent evaluation tracking. A unified platform that addresses these problems in one place would reduce coordination overhead, improve reproducibility, and make it easier to reason about data and model quality across the organization.
