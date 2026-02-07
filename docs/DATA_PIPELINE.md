# Data Pipeline

Internal documentation for the LLM Data Platform data pipeline. Describes how raw data is ingested, cleaned, filtered, versioned, and built from config.

---

## 1. Overview

The data pipeline turns raw input files (exports, annotations, third-party data) into versioned datasets that downstream training and evaluation can rely on. It addresses:

- **Inconsistent formats**: Multiple sources and ad hoc exports; no single schema.
- **Unversioned data**: Hard to know what data a given run used or to reproduce a past dataset.
- **Unreproducible builds**: Re-running the same steps should yield the same artifact.

Dataset versioning is central. Every built dataset is written to a named version directory with a content hash. That hash is stored in metadata so you can confirm a dataset matches a previous build and so training runs can record exactly which dataset (and which hash) they used. Reproducibility and auditing depend on this.

---

## 2. Ingestion layer

Ingestion loads a file and normalizes it into a single in-memory structure. No cleaning or filtering is done at this stage.

**Supported formats**

- **JSON**: File must be a JSON array. Each element must be an object with `"input"` and `"output"` string keys. Format is detected by `.json` extension.
- **CSV**: Header row expected. Prefers columns named `input` and `output` (case-insensitive); if missing, the first two columns are used as input and output. Format is detected by `.csv` extension.
- **Plain text**: One sample per line. If a line contains a tab, the line is split on the first tab into input (before) and output (after). Otherwise the full line is input and output is empty. Format is detected by `.txt` or `.text` extension.

**Normalized sample structure**

Every ingested row becomes a `Sample` with four fields:

| Field   | Type | Description |
|--------|------|-------------|
| `id`   | str  | Unique per sample within the run; format `{source}_{index}`. |
| `input`| str  | Normalized input text. |
| `output`| str | Normalized output text. |
| `source`| str | Label passed at ingestion (e.g. dataset or system name). |

The ingestion entrypoint is `load_dataset(path, source)` in `src/data/ingestion.py`. It picks the loader from the file extension and returns a list of `Sample` objects.

---

## 3. Cleaning and filtering

Cleaning and filtering are rule-based and deterministic. Same input and config always produce the same result.

**Cleaning (deterministic rules)**

- **Empty samples**: A sample is dropped if `input` or `output` is missing or blank (empty or whitespace-only). Implemented in `remove_empty_samples`.
- **Duplicates**: Duplicates are defined by the pair `(input, output)`. The first occurrence is kept; later ones are dropped. Implemented in `remove_duplicate_samples`.

**Filtering (deterministic rules)**

- **Minimum length**: Samples where either `input` or `output` has fewer than `min_length` characters are dropped. Config key: `min_length`.
- **Noise (optional)**: Samples where either field has the same character repeated more than a threshold (e.g. 10) are dropped. Config keys: `filter_noise`, `noise_max_repeat`.

The combined step is `clean_and_filter(samples, config)` in `src/data/filtering.py`: it runs cleaning first (empty, then duplicates if enabled), then length and noise filtering according to config.

**Why no LLMs here**

Cleaning and filtering are kept deterministic and fast. They use only string rules, set membership, and length checks. LLMs are not used so that:

- Runs are reproducible and independent of model availability or non-determinism.
- There are no external calls or cost.
- Behavior is easy to reason about and debug.

Semantic deduplication or quality scoring can be handled later in separate tooling if needed.

---

## 4. Dataset versioning

Versioning writes a list of samples to a fixed layout and records metadata (including a content hash) so the dataset is identifiable and verifiable.

**Folder structure**

Each version is a single directory under a base output path (default `artifacts/datasets`):

```
artifacts/datasets/
  <version_name>/
    data.jsonl
    metadata.json
```

`version_name` is chosen in config (e.g. `support_v1`). Parent directories are created if missing.

**data.jsonl**

One JSON object per line. Each object has the same keys as a `Sample`: `id`, `input`, `output`, `source`. Keys are emitted in a fixed order so that the file content is canonical for hashing.

**metadata.json**

A single JSON object with:

- `dataset_version`: The version name.
- `num_samples`: Number of lines in `data.jsonl`.
- `config`: The full dataset config used for the build (ingestion path, source, cleaning/filter options, etc.).
- `dataset_hash`: SHA256 hash of the dataset content (see below).

**Dataset hashing**

The dataset hash is computed from the exact content that is written to `data.jsonl`: each sample serialized as one line of JSON with sorted keys, UTF-8 encoded. Order of samples is preserved. Same list of samples in the same order yields the same hash. This allows:

- Storing which dataset a training run used by recording the version name and hash.
- Checking that a copy or re-build matches the original by recomputing the hash and comparing to `metadata.json`.

The function `compute_dataset_hash(samples)` in `src/data/versioning.py` implements this. `create_dataset_version` writes the files and stores the hash in `metadata.json`.

---

## 5. Config-driven pipeline

Dataset builds are driven by YAML config files under `configs/datasets/`. The pipeline reads one config file and runs ingestion, cleaning, filtering, and versioning from it. No build logic is hardcoded in code; all choices come from the config.

**Config fields that control ingestion**

- `source` (required): Label passed to `load_dataset`; used as the `source` field on every sample and in sample IDs.
- `input_path` (required): Path to the raw file (JSON, CSV, or text). Relative paths are relative to the process current working directory.

**Config fields that control cleaning and filtering**

- `remove_duplicates` (optional, bool): If true, duplicate (input, output) pairs are removed after removing empty samples.
- `min_length` (optional, int): Drop samples where either input or output has fewer than this many characters.
- `filter_noise` (optional, bool): If true, drop samples with excessive repeated characters in input or output.
- `noise_max_repeat` (optional, int): Used when `filter_noise` is true; default 10.

**Config fields for versioning**

- `version_name` (required): Name of the version directory (e.g. `support_v1`).
- `output_dir` (optional): Base directory for version folders; default `artifacts/datasets`.

The pipeline does not add or remove config keys; the full config is written into `metadata.json` so the exact build is documented.

---

## 6. How to build a dataset

1. Add or edit a YAML file under `configs/datasets/` with the fields above (at least `source`, `input_path`, `version_name`).
2. Run the pipeline by calling the build function with that config path.

Example config (`configs/datasets/support_dataset.yaml`):

```yaml
source: support
input_path: data/raw/support_export.csv
min_length: 10
remove_duplicates: true
version_name: support_v1
```

Example build (from project root, with `src` on the Python path):

```python
from src.data.pipeline import build_dataset_from_config

version_path = build_dataset_from_config("configs/datasets/support_dataset.yaml")
# version_path is e.g. "artifacts/datasets/support_v1"
```

Or from the command line, after ensuring the working directory and path are set:

```bash
python -c "from src.data.pipeline import build_dataset_from_config; build_dataset_from_config('configs/datasets/support_dataset.yaml')"
```

The pipeline logs each step (config load, ingestion path, sample counts, version path). The function returns the path to the created version directory.

---

## 7. Dataset artifact structure

After a successful build, the artifact for version `support_v1` looks like this:

```
artifacts/datasets/support_v1/
  data.jsonl      # One JSON object per line: id, input, output, source
  metadata.json   # dataset_version, num_samples, config, dataset_hash
```

- **data.jsonl**: Use for training or evaluation; each line is one `Sample` as JSON.
- **metadata.json**: Use to see how the dataset was built (`config`), how many samples it has (`num_samples`), and to verify or reference the content (`dataset_hash`, `dataset_version`).

Downstream systems should record at least `dataset_version` and `dataset_hash` when consuming a dataset so runs are traceable and reproducible.
