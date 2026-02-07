"""
Config-driven dataset pipeline for the LLM Data Platform.
"""

from __future__ import annotations

from pathlib import Path

from . import ingestion
from . import filtering
from . import versioning


def _load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for pipeline. Install with: pip install pyyaml")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset_from_config(config_path: str) -> str:
    """
    Run the full dataset pipeline from a YAML config file.
    Returns the path to the created dataset version directory.
    """
    print(f"[pipeline] Loading config from {config_path}")
    config = _load_yaml_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML object (dict)")

    input_path = config["input_path"]
    source = config["source"]
    print(f"[pipeline] Ingesting from {input_path} (source={source})")
    samples = ingestion.load_dataset(input_path, source)
    print(f"[pipeline] Loaded {len(samples)} samples")

    # Build clean/filter config from dataset config (only keys that clean_and_filter uses)
    filter_config = {}
    if "min_length" in config:
        filter_config["min_length"] = config["min_length"]
    if "remove_duplicates" in config:
        filter_config["remove_duplicates"] = config["remove_duplicates"]
    if config.get("filter_noise"):
        filter_config["filter_noise"] = True
        if "noise_max_repeat" in config:
            filter_config["noise_max_repeat"] = config["noise_max_repeat"]

    print("[pipeline] Applying cleaning and filtering")
    samples = filtering.clean_and_filter(samples, filter_config)
    print(f"[pipeline] After clean/filter: {len(samples)} samples")

    version_name = config["version_name"]
    output_dir = config.get("output_dir", "artifacts/datasets")
    print(f"[pipeline] Creating dataset version '{version_name}' in {output_dir}")
    version_path = versioning.create_dataset_version(
        samples,
        version_name=version_name,
        config=config,
        output_dir=output_dir,
    )
    print(f"[pipeline] Done. Output: {version_path}")
    return version_path
