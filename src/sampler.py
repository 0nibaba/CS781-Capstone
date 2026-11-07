import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tokenizers import Tokenizer
from train import MemmapDataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tok = Tokenizer.from_file(f"{ROOT}/checkpoints/tokenizers/legal_32000.json")
dl = MemmapDataLoader(f"{ROOT}/training_data/tokens", block_size=1024)

X, Y = dl.get_batch("train", batch_size=4, device="cpu")

print(f"\nX shape {tuple(X.shape)}  dtype {X.dtype}")
print(f"Y shape {tuple(Y.shape)}")
print(f"max token id: {X.max().item()}  (must be < 32000)")

shifted_ok = (X[:, 1:] == Y[:, :-1]).all().item()
print(f"targets shifted by one: {shifted_ok}")

print("\n--- first sequence, first 120 tokens ---")
print(tok.decode(X[0][:120].tolist()))

print("\n--- does it contain document separators? ---")
eot = tok.token_to_id("<|endoftext|>")
for i in range(4):
    n = (X[i] == eot).sum().item()
    print(f"  seq {i}: {n} separators in {X.shape[1]} tokens")

print("\n--- val batch ---")
Xv, Yv = dl.get_batch("val", batch_size=2, device="cpu")
print(tok.decode(Xv[0][:120].tolist()))