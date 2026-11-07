import os, json
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training_data/raw/Phase_4_CJPE/Train")
TEST_DIR = os.path.join(ROOT, "test_data/Phase_4_CJPE/Test")
TOKENIZER = os.path.join(ROOT, "checkpoints/tokenizers/legal_32000.json")

MAX_LEN = 1024
PAD_ID = 0
MODES = ("head", "tail", "head_tail")


def load_split(split):
    path = (os.path.join(TEST_DIR, "test.json") if split == "test"
            else os.path.join(TRAIN_DIR, f"{split}.json"))
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "text": v}
             for k, v in d.items()]
    out = []
    for x in d:
        rec = {"id": x["id"], "text": x["text"]}
        if split != "test" and x.get("label") is not None:
            rec["label"] = int(x["label"])
        out.append(rec)
    return out


def select(ids, mode, max_len=MAX_LEN):
    if len(ids) <= max_len:
        return ids
    if mode == "head":
        return ids[:max_len]
    if mode == "tail":
        return ids[-max_len:]
    if mode == "head_tail":
        h = max_len // 2
        return ids[:h] + ids[-(max_len - h):]
    raise ValueError(mode)


class CJPEDocs:
    def __init__(self, docs, tok, mode="tail", max_len=MAX_LEN, batch=256):
        assert mode in MODES, mode
        self.items = []
        for i in range(0, len(docs), batch):
            part = docs[i:i + batch]
            encs = tok.encode_batch([d["text"] for d in part])
            for d, e in zip(part, encs):
                self.items.append({
                    "id": d["id"],
                    "input_ids": select(e.ids, mode, max_len),
                    "label": d.get("label", -1),
                    "n_tokens": len(e.ids),
                })

    def __len__(self):
        return len(self.items)


def collate(chunk, device):
    n = max(len(it["input_ids"]) for it in chunk)
    ids = torch.full((len(chunk), n), PAD_ID, dtype=torch.long)
    msk = torch.zeros(len(chunk), n, dtype=torch.long)
    lab = torch.tensor([it["label"] for it in chunk], dtype=torch.long)
    for i, it in enumerate(chunk):
        L = len(it["input_ids"])
        ids[i, :L] = torch.tensor(it["input_ids"], dtype=torch.long)
        msk[i, :L] = 1
    return ids.to(device), msk.to(device), lab.to(device)


def batches(ds, batch_size, device, shuffle=True, seed=0):
    idx = list(range(len(ds)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
    # length-sorted buckets cut padding waste without breaking shuffling
    for i in range(0, len(idx), batch_size):
        chunk = [ds.items[j] for j in idx[i:i + batch_size]]
        yield chunk, collate(chunk, device)