"""Download Phase 1 data: train -> data/raw/, test -> test_data/."""

import os
from huggingface_hub import hf_hub_download

REPO = "Exploration-Lab/CS-781-Capstone"
HF_TOKEN = os.environ.get("HF_TOKEN")   # None is fine if you ran huggingface-cli login

BASE = "/data1/yashy26"
TRAIN_DEST = f"{BASE}/training_data/raw"
TEST_DEST = f"{BASE}/test_data"

TRAIN_FILES = [
    "Phase_1_Pretraining/Train/train_acts.json",
    "Phase_1_Pretraining/Train/train_high_courts.json",
    "Phase_1_Pretraining/Train/train_supreme_court.json",
    "Phase_1_Pretraining/Train/train_tribunals.json",
]
TEST_FILES = [
    "Phase_1_Pretraining/Test/test.json",
]

for dest, files in [(TRAIN_DEST, TRAIN_FILES), (TEST_DEST, TEST_FILES)]:
    os.makedirs(dest, exist_ok=True)
    for path in files:
        print(f"downloading {path} ...", flush=True)
        fp = hf_hub_download(
            repo_id=REPO,
            filename=path,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=dest,
        )
        print(f"  -> {fp}  ({os.path.getsize(fp) / 1e9:.2f} GB)\n", flush=True)

print("done")