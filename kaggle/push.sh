#!/bin/zsh
# Upload the code and one dataset version as a private Kaggle dataset, then start the GPU run.
# Usage: KAGGLE_USER=<name> KAGGLE_BIN=<kaggle command> kaggle/push.sh <dataset version folder>
set -e
ROOT=${0:A:h:h}
DATA=${1:?give a dataset version folder}
USER=${KAGGLE_USER:?set KAGGLE_USER}
K=${KAGGLE_BIN:-kaggle}
B=$(mktemp -d)

mkdir -p "$B/data/code/src" "$B/data/code/dataset"
cp -R "$ROOT/src/finetune" "$B/data/code/src/"
touch "$B/data/code/src/__init__.py"
cp "$DATA"/{train,validation,test}.jsonl "$DATA/metadata.json" "$B/data/code/dataset/"
find "$B/data/code" -name __pycache__ -prune -exec rm -rf {} +
cat > "$B/data/dataset-metadata.json" <<EOF
{"title": "llm-pipeline-code", "id": "$USER/llm-pipeline-code", "licenses": [{"name": "MIT"}]}
EOF
if "$K" datasets status "$USER/llm-pipeline-code" >/dev/null 2>&1; then
  "$K" datasets version -p "$B/data" -m "update" --dir-mode zip
else
  "$K" datasets create -p "$B/data" --dir-mode zip
fi

mkdir -p "$B/kernel"
cp "$ROOT/kaggle/run.py" "$B/kernel/run.py"
cat > "$B/kernel/kernel-metadata.json" <<EOF
{
  "id": "$USER/llm-pipeline-train",
  "title": "llm-pipeline-train",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER/llm-pipeline-code"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
"$K" kernels push -p "$B/kernel"
