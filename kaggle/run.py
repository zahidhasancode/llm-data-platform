"""Kaggle entry point: install pinned libraries, train, evaluate, keep the outputs."""

import glob
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
# Kaggle may attach two T4s; a 4-bit model cannot be split by DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(list(cmd))


def find_code() -> str:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.unpack_archive(z, f"{WORK}/unzipped")
    hits = glob.glob("/kaggle/input/**/src/finetune/train.py", recursive=True)
    hits += glob.glob(f"{WORK}/unzipped/**/src/finetune/train.py", recursive=True)
    if not hits:
        raise SystemExit("code not found")
    return os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))


sh(sys.executable, "-m", "pip", "install", "-q",
   "transformers==4.46.3", "peft==0.13.2", "trl==0.12.2", "accelerate==1.1.1",
   "bitsandbytes==0.49.2", "datasets==3.2.0")

shutil.copytree(find_code(), f"{WORK}/repo", dirs_exist_ok=True)
os.chdir(f"{WORK}/repo")
sh("nvidia-smi")
EPOCHS = "1.0"
sh(sys.executable, "-m", "src.finetune.train", "--data", "dataset", "--epochs", EPOCHS)
sh(sys.executable, "-m", "src.finetune.evaluate", "--data", "dataset")

out = f"{WORK}/outputs"
shutil.copytree("results", f"{out}/results", dirs_exist_ok=True)
shutil.copytree("outputs/adapter", f"{out}/adapter", dirs_exist_ok=True)
print("finished; outputs in", out)
