# Training Pipeline

Internal documentation for the LLM Data Platform training pipeline. Describes configuration, training, registration, evaluation, and the end-to-end flow from config to evaluation results.

---

## 1. Overview

The training pipeline turns a training config and a versioned dataset into a registered model and an evaluation result. It does not run actual deep learning; it is a platform layer that loads configs, resolves datasets, records model metadata, and produces evaluation artifacts so that runs are traceable and reproducible.

Reproducible training matters because models must be tied to the exact dataset version, hyperparameters, and code that produced them. Without that, you cannot re-run a training job to verify a result, compare runs, or audit what a deployed model was trained on. The pipeline records dataset version, full training config, and sample count in model metadata and uses fixed paths so that the same config and model version always follow the same steps.

---

## 2. Training configuration system

Training runs are driven by YAML config files, typically under `configs/training/`. Each file defines the base model, dataset version, and training hyperparameters.

**YAML configs**

Config files are loaded with `load_training_config(path)` from `src/training/config.py`. The loader expects a single YAML object (key-value map). Required keys are validated; missing or invalid values raise `ValueError`.

**TrainingConfig structure**

The loaded config is parsed into a `TrainingConfig` dataclass with these fields:

| Field            | Type  | Description |
|------------------|-------|-------------|
| `base_model`     | str   | Identifier of the base model (e.g. mistral-7b). |
| `dataset_version`| str   | Name of the dataset version folder under `artifacts/datasets/`. |
| `learning_rate`  | float | Learning rate (numeric; stored as float). |
| `epochs`         | int   | Number of epochs (must be >= 1). |
| `batch_size`     | int   | Batch size (must be >= 1). |

Validation rules: `base_model` and `dataset_version` must be non-empty after stripping whitespace; `learning_rate` can be int or float and is converted to float; `epochs` and `batch_size` must be positive integers. The same YAML file always yields the same `TrainingConfig`.

---

## 3. Trainer module

The trainer lives in `src/training/trainer.py`. It uses the training config to locate the dataset and write model metadata; it does not perform actual model training.

**How datasets are used**

The trainer expects versioned datasets under `artifacts/datasets/`. The config’s `dataset_version` is the name of a subdirectory (e.g. `dataset_v1`). The trainer resolves the path as `artifacts/datasets/<dataset_version>/` and expects that directory to contain `data.jsonl`. It does not read the data for training; it only counts lines in `data.jsonl` to get the number of training samples. That count is stored in model metadata. If the dataset directory or `data.jsonl` is missing, the trainer raises `FileNotFoundError`.

**How model metadata is generated**

After resolving the dataset and counting samples, the trainer creates a model version directory under `artifacts/models/` (or a custom `artifacts_dir`). The directory name is the `model_version` passed by the caller (e.g. `model_v1`). Inside that directory it writes `metadata.json` with:

- `model_version`: The version string for this run.
- `base_model`: From config.
- `dataset_version`: From config.
- `training_config`: The full config as a dict (e.g. via `asdict(config)`).
- `num_training_samples`: The count from `data.jsonl`.

So every model artifact is tied to a specific dataset version, config, and sample count. The function returns the path to the model version directory.

---

## 4. Model registry

The registry provides a single list of all models that have been through the pipeline, so you can list them and look up metadata by model version without scanning the filesystem.

**Purpose**

The registry stores a copy of each model’s metadata (the same structure as `metadata.json`) when a model is registered. It is used to:

- List all registered models.
- Fetch metadata for a given model version by name.

The registry does not replace the per-model `metadata.json`; it duplicates the metadata for lookup. Path and storage are controlled by the caller (e.g. the end-to-end pipeline).

**registry.json structure**

The registry file (by default `artifacts/models/registry.json`) is a single JSON object:

```json
{
  "models": [
    {
      "model_version": "model_v1",
      "dataset_version": "dataset_v1",
      "base_model": "mistral-7b",
      "num_training_samples": 45231,
      "training_config": { ... }
    }
  ]
}
```

Each entry in `models` is the full metadata dict that was written to that model’s `metadata.json`. New models are appended. If the file does not exist, it is created when the first model is registered. Functions: `register_model(metadata, registry_path)`, `list_models(registry_path)`, `get_model(model_version, registry_path)`.

---

## 5. Evaluation module

The evaluation module lives in `src/evaluation/evaluator.py`. It reads model metadata, computes or simulates metrics, and writes an evaluation artifact. No real model inference is run; metrics are deterministic and derived from model version and dataset version.

**What metrics are stored**

Each evaluation produces a JSON file with:

- `model_version`: The model that was evaluated.
- `dataset_version`: Taken from the model’s metadata.
- `metrics`: A dict with:
  - `quality_score` (float): Simulated quality metric.
  - `latency_ms` (int): Simulated latency in milliseconds.
  - `cost_per_1k_tokens` (float): Simulated cost.

The exact semantics of these metrics depend on how the evaluator is implemented; in the current implementation they are derived deterministically from the model and dataset version strings so that the same inputs always yield the same evaluation file.

**Where evaluation results live**

Results are written under `artifacts/evaluations/` (or a custom `output_dir`). For a given `model_version`, the evaluator creates a directory `artifacts/evaluations/<model_version>/` and writes `evaluation.json` there. So one evaluation artifact per model version, and the path is predictable from the version name.

---

## 6. End-to-end pipeline

The end-to-end pipeline is implemented in `src/training/pipeline.py`. A single function runs training, registration, and evaluation in order and returns the evaluation result.

**Steps from training config to evaluation**

1. **Train**: `train_model(training_config_path, model_version)` is called with default `artifacts_dir` of `artifacts/models`. This loads the config, finds the dataset at `artifacts/datasets/<dataset_version>/`, counts samples in `data.jsonl`, runs the (simulated) training step, and writes `artifacts/models/<model_version>/metadata.json`.

2. **Load metadata**: The pipeline reads `metadata.json` from the new model version directory via `load_model_metadata(model_version)`.

3. **Register**: The pipeline calls `register_model(metadata, registry_path)` with the metadata dict and the default registry path `artifacts/models/registry.json`. The model is appended to the registry.

4. **Evaluate**: The pipeline calls `evaluate_model(model_version)` with default `models_dir` and `output_dir`. The evaluator reads the same metadata, produces metrics, and writes `artifacts/evaluations/<model_version>/evaluation.json`.

5. **Return results**: The pipeline loads `evaluation.json` and returns its contents as a dict (e.g. `model_version`, `dataset_version`, `metrics`).

All paths are fixed so that the same config and model version produce the same artifact locations and the same evaluation output. No external services are called.

---

## 7. Artifact structure

Artifacts live under three top-level directories. Paths are relative to the project or process working directory unless overridden.

**datasets/**

- Path: `artifacts/datasets/`
- Contents: One subdirectory per dataset version (e.g. `artifacts/datasets/dataset_v1/`).
- Each version directory contains:
  - `data.jsonl`: One JSON object per line (training data).
  - `metadata.json`: Dataset version name, sample count, config, dataset hash.
- Produced by the data pipeline (`build_dataset_from_config`), not by the training pipeline. The training pipeline only reads from here (e.g. to count samples and resolve dataset version).

**models/**

- Path: `artifacts/models/`
- Contents:
  - One subdirectory per model version (e.g. `artifacts/models/model_v1/`), each with:
    - `metadata.json`: model_version, base_model, dataset_version, training_config, num_training_samples.
  - `registry.json`: List of all registered model metadata entries.
- Model version directories and metadata are produced by the trainer; the registry is updated by the pipeline (or any caller that invokes `register_model`).

**evaluations/**

- Path: `artifacts/evaluations/`
- Contents: One subdirectory per evaluated model version (e.g. `artifacts/evaluations/model_v1/`), each with:
  - `evaluation.json`: model_version, dataset_version, metrics (e.g. quality_score, latency_ms, cost_per_1k_tokens).
- Produced by the evaluation module when `evaluate_model` is run (e.g. from the end-to-end pipeline).

---

## 8. How to run the pipeline

Run the full pipeline (train, register, evaluate) by calling `run_training_pipeline` with a path to a training config and a model version string. Ensure the dataset version referenced in the config already exists under `artifacts/datasets/` (built via the data pipeline).

**Example function call**

```python
from src.training.pipeline import run_training_pipeline

results = run_training_pipeline(
    training_config_path="configs/training/mistral_finetune.yaml",
    model_version="model_v1",
)
# results is the dict from evaluation.json, e.g.:
# {"model_version": "model_v1", "dataset_version": "dataset_v1", "metrics": {...}}
```

**Example script**

From the project root, with `src` on the Python path:

```bash
python -c "
from src.training.pipeline import run_training_pipeline
r = run_training_pipeline('configs/training/mistral_finetune.yaml', 'model_v1')
print(r)
"
```

The pipeline creates the model directory and registry file if they do not exist. It expects the training config to be valid and the dataset version folder to be present; otherwise it raises (e.g. `FileNotFoundError`, `ValueError`).
