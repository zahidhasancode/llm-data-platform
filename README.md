# LLM Data Platform

A small Python prototype for preparing fine-tuning data for LLMs: it loads raw Q&A files, cleans them with fixed rules, and writes versioned datasets with a content hash. A second stage records training runs in a model registry, but training and evaluation are simulated.

## Status

This is a **prototype**, not a working training system.

| Part | State |
| --- | --- |
| Data ingestion (JSON, CSV, tab-separated text) | Works, tested |
| Cleaning and filtering (empty, duplicate, length, noise rules) | Works, tested |
| Dataset versioning with SHA-256 content hash | Works, tested |
| Config-driven dataset build from YAML | Works, tested |
| Training config loading and validation | Works, tested |
| Model registry (JSON file) | Works, tested |
| Training | **Simulated.** No model is downloaded or trained. Only metadata is written. |
| Evaluation | **Simulated.** Metrics are derived from a hash of the version names. They are not measurements. |
| `src/agent/`, `src/deployment/` | Empty placeholders |

## What it does

1. Reads a raw file of `input` / `output` pairs (for example a support Q&A export).
2. Drops empty pairs, exact duplicates, pairs that are too short, and pairs with long runs of one character.
3. Writes the result as a named dataset version: `data.jsonl` plus `metadata.json` with the sample count, the full build config and a SHA-256 hash of the data.
4. Given a training config that names a dataset version, writes model metadata (base model, dataset version, hyperparameters, sample count), adds it to a JSON registry, and writes an evaluation file.

Every step is deterministic. The same input and config produce byte-identical output files.

## How it works

```
raw file ──> ingestion ──> cleaning + filtering ──> versioning ──> artifacts/datasets/<version>/
             (Sample)                                (hash)              data.jsonl, metadata.json

training config ──> trainer (simulated) ──> registry ──> evaluator (simulated)
                    artifacts/models/<model>/  registry.json  artifacts/evaluations/<model>/
```

| Module | Role |
| --- | --- |
| `src/data/ingestion.py` | `load_dataset(path, source)` picks a loader by file extension and returns `Sample(id, input, output, source)` objects. IDs are `<source>_<row index>`. |
| `src/data/cleaning.py` | `remove_empty_samples`, `remove_duplicate_samples` (exact `(input, output)` match, first one kept). |
| `src/data/filtering.py` | `filter_by_min_length`, `filter_noise` (same character repeated more than `noise_max_repeat` times), and `clean_and_filter`, which runs the rules in order from a config dict. |
| `src/data/versioning.py` | `compute_dataset_hash` (SHA-256 over the exact lines of `data.jsonl`) and `create_dataset_version`. |
| `src/data/pipeline.py` | `build_dataset_from_config(yaml_path)` runs the four steps above from a YAML file in `configs/datasets/`. |
| `src/training/config.py` | `load_training_config` parses and validates `configs/training/*.yaml` into a `TrainingConfig` dataclass. |
| `src/training/trainer.py` | `train_model` checks the dataset version exists, counts its samples and writes `metadata.json`. The training step is a no-op. |
| `src/training/registry.py` | `register_model`, `list_models`, `get_model` on a JSON file. Registering the same model version again replaces its entry. |
| `src/evaluation/evaluator.py` | `evaluate_model` writes `evaluation.json` with `quality_score`, `latency_ms` and `cost_per_1k_tokens` computed from a SHA-256 of `model_version:dataset_version`. |
| `src/training/pipeline.py` | `run_training_pipeline(config_path, model_version)` runs train, register and evaluate, and returns the evaluation dict. |

More detail is in [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) and [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md).

## Quick start

Needs Python 3.11. These are the commands used to check this README, run from a fresh clone on macOS.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Lint and tests
ruff check .
pytest

# Build the toy dataset (examples/support_sample.csv, nine hand-written rows)
python -c "from src.data.pipeline import build_dataset_from_config; build_dataset_from_config('configs/datasets/example_support.yaml')"

# Run the simulated train -> register -> evaluate flow on it
python -c "from src.training.pipeline import run_training_pipeline; print(run_training_pipeline('configs/training/example_finetune.yaml', 'model_v1'))"
```

Output goes to `artifacts/` in the current directory. That folder is git-ignored.

## Example output

The dataset build prints:

```
[pipeline] Loading config from configs/datasets/example_support.yaml
[pipeline] Ingesting from examples/support_sample.csv (source=support_example)
[pipeline] Loaded 9 samples
[pipeline] Applying cleaning and filtering
[pipeline] After clean/filter: 5 samples
[pipeline] Creating dataset version 'support_example_v1' in artifacts/datasets
[pipeline] Done. Output: artifacts/datasets/support_example_v1
```

The four dropped rows are a duplicate, a question with no answer, a too-short pair (`Hi` / `Hello!`) and a row with a long run of one letter.

`artifacts/datasets/support_example_v1/metadata.json`:

```json
{
  "dataset_version": "support_example_v1",
  "num_samples": 5,
  "config": {
    "source": "support_example",
    "input_path": "examples/support_sample.csv",
    "min_length": 10,
    "remove_duplicates": true,
    "filter_noise": true,
    "noise_max_repeat": 10,
    "version_name": "support_example_v1"
  },
  "dataset_hash": "31668bb8be6f2930553b27d7c2db27284e4e6b3dbe0e5e0f48f7eb9488386bbe"
}
```

One line of `data.jsonl`:

```json
{"id": "support_example_2", "input": "Can I change the email on my account?", "output": "Yes. Go to Settings, then Profile, and edit the email field. You will need to confirm the new address.", "source": "support_example"}
```

The training pipeline prints:

```
{'model_version': 'model_v1', 'dataset_version': 'support_example_v1', 'metrics': {'quality_score': 0.8342, 'latency_ms': 80, 'cost_per_1k_tokens': 0.018}}
```

These metric values are **not** results. They come from a hash of the strings `model_v1` and `support_example_v1`, so they change when you rename a version and say nothing about any model.

## Tests and CI

`pytest` runs 64 tests in under a second. They cover:

- ingestion of JSON, CSV and text files, including null fields, column selection and bad input;
- each cleaning and filtering rule and their order;
- hash stability, its link to the bytes of `data.jsonl`, and byte-identical rebuilds;
- training config validation, including the shipped example configs;
- the registry file;
- both pipelines end to end on the toy example.

No test needs a GPU, a network connection or a model download, because nothing in the code uses one.

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check .` and `pytest` on Python 3.11 for pushes to `main` and for pull requests.

## Limitations

- **No real training.** `_simulate_training` in `src/training/trainer.py` does nothing. There is no model loading, tokenisation, fine-tuning or checkpoint.
- **No real evaluation.** The evaluator never runs a model. Its numbers are placeholders.
- Cleaning is rule-based only: exact-match duplicates, character length, repeated characters. No near-duplicate detection, language or quality checks, or PII removal.
- Files are loaded fully into memory. Fine for small exports, not for large corpora.
- Paths such as `artifacts/datasets` are relative to the working directory, and the trainer always reads datasets from `artifacts/datasets`.
- There is no command-line interface; the pipelines are called from Python.
- `configs/datasets/support_dataset.yaml` and `configs/training/mistral_finetune.yaml` point at data (`data/raw/support_export.csv`, `dataset_v1`) that is not in the repo. Use the `example_*` configs to try it.

## Next steps

- Replace the simulated trainer with a real fine-tuning step (for example a small LoRA run with Hugging Face `transformers` and `peft`), keeping the config and registry contract.
- Replace the simulated metrics with a real evaluation on a held-out split of the dataset version.
- Record the dataset hash, not only the version name, in the model metadata.
- Add a small CLI (`build-dataset`, `train`) instead of `python -c`.

## Tech used

Python 3.11, standard library (`csv`, `json`, `hashlib`, `dataclasses`, `pathlib`, `re`), PyYAML, pytest, ruff, GitHub Actions.

## Licence

MIT. See `LICENSE`.
