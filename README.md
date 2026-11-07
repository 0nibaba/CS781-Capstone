# CS781 Capstone — In-House Legal Intelligence System

A legal NLP stack built from a single 110M-parameter transformer pretrained
**from scratch** on 3.5B tokens of Indian legal text, then adapted to six
downstream tasks. No pretrained language model, no external legal corpus and no
external LLM is used at any point: every downstream model is initialised from a
Phase-1 checkpoint trained in this repository.

| Phase | Task | Metric | Best score | Best system |
|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | **0.8343** | prefix-LM, 3B tokens |
| 2 | Legal NER | strict macro-F1 | **0.5683** | decoder + CRF, all-data retrain |
| 3 | Rhetorical role prediction | macro-F1 | **0.5023** | decoder, marker pooling |
| 4 | Court judgment prediction | macro-F1 | **0.6997** | full FT, tail-1024, prior-corrected |
| 5 | Legal statute identification | macro-F1 | **0.3978** | 5-model ensemble, per-label thresholds |
| 6 | Abstractive summarization | BERTScore F1 | **0.8260** | distilled 6/12-layer prefix-LM student |
| 7 | Prior case retrieval | micro-F1@7 | **0.2965** | reciprocal-rank fusion of 4 dense rankers |

The full experimental record, including predictions registered *before* runs and
the eight that were refuted, is in [`DECISIONS.md`](logs/DECISIONS.md).

---

## Contents

1. [Architecture](#1-architecture)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Phase 1 — pretraining](#4-phase-1--pretraining)
5. [Phases 2–7 — downstream adaptation](#5-phases-27--downstream-adaptation)
6. [Reproducing the best submission per phase](#6-reproducing-the-best-submission-per-phase)
7. [Verification suite](#7-verification-suite)
8. [Known issues and gotchas](#8-known-issues-and-gotchas)

---

## 1. Architecture

### 1.1 Base model

A 12-layer, 12-head, `d_model=768` pre-norm decoder — **110,316,288
parameters**, inside the 50–300M competition bound:

- **RMSNorm** computed in fp32 (normalisation is where fp16/bf16 loses precision)
- **GELU** (tanh approximation), 4× MLP expansion
- **Learned absolute positions**, block size 1024 — this is the hard ceiling that
  shapes every downstream representation decision
- **No bias terms** anywhere (`Linear(bias=False)`, RMSNorm has scale only).
  Consequence: **BitFit is inapplicable to this architecture**, discovered when a
  BitFit arm trained 1,538 parameters and silently became a head-only probe.
- **Tied input/output embeddings** — frees 24.6M parameters, roughly two extra layers
- Residual projections scaled by `1/sqrt(2 * n_layer)`
- Attention via `scaled_dot_product_attention` (FlashAttention when available)

### 1.2 The four Phase-1 architectures

All trained at a **matched 1.494B-token budget** so the comparison is fair.

| Architecture | File | Objective |
|---|---|---|
| Decoder | `decoder_only_transformer.py` | causal LM |
| Prefix-LM | `prefix_lm.py`, `prefix_gpt.py` | bidirectional over prompt sampled from `[0.7T, 0.9T]`, causal over continuation |
| Efficient (windowed) | `efficient_attn.py` | local–global attention, window 256 + every 4th layer global |
| Encoder–decoder | `encoder_decoder.py`, `span_corruption.py` | T5-style span corruption, 4 encoder / 6 decoder layers |

The encoder–decoder needed a **long-terminal corruption mode** added before
training: a 15% corruption budget over 1024 tokens yields ~150 masked tokens
across ~6 spans (measured p99 span length 78), so no span approaches the ~166
tokens Phase 1's evaluation requires. 15% of examples instead carry one
120–200 token span at the sequence end.

### 1.3 Tokenizer

Byte-level BPE, 32k vocabulary, trained on a 350MB sample reweighted toward Acts
and Supreme Court text (`train_tokenizer.py`).

- Fertility 1.296 vs GPT-2's 1.353 on legal text; 48k buys a further 1.3% for
  12.3M extra embedding parameters — a poor trade against a fixed budget
- **Explicit digit splitting was tested and rejected**: it *cost* 7.6% fertility,
  because byte-level BPE already splits digits
- `<|endoftext|>` is id 0; `<|unused0..9|>` are reserved and later used as
  sentence separators and domain markers in Phase 3

### 1.4 Downstream heads

All share the Phase-1 backbone via `load_base()` in `ner_model.py`.

| Phase | Head | File |
|---|---|---|
| 2 | token classification + optional linear-chain CRF | `ner_model.py`, `crf.py` |
| 3 | separator-token ("marker") pooling + optional CRF | `rr_model.py` |
| 4 | pooled document classification (last / mean / attention) | `cjpe_model.py` |
| 5 | 100 independent sigmoids (binary relevance) | `lsi_model.py` |
| 6 | encoder–decoder or prompt-conditioned continuation | `summ_model.py` |
| 7 | L2-normalised bi-encoder, max-sim over chunks | `pcr_model.py` |

### 1.5 Adaptation techniques (all implemented from scratch)

- **Full fine-tuning**
- **Partial / layer-wise fine-tuning** — top-k transformer blocks only
- **LoRMA** (`lorma.py`) — multiplicative where LoRA is additive:
  `h = I(BA) W_0 x` with `I(BA) = (alpha/r) BA + I_d`, so the adapter starts as
  the identity. The codebase uses a **fused `c_attn` of shape (d, 3d)**, which the
  paper does not anticipate; applying one transform to the fused output would
  adapt Q, K and V jointly under a shared (3d × 3d) map, which is a different
  method. This implementation slices and applies **separate transforms to the Q
  and V thirds, leaving K untouched**.
- **Knowledge distillation** (`distill_*.py`) — every experiment runs **two arms
  from one student initialisation**: teacher-KL + hard labels, versus hard labels
  alone. Without the control arm the gain cannot be attributed to the teacher.
- **Head-only linear probe** — isolates how much comes from adapting the encoder

---

## 2. Repository layout

```
src/
├── Phase 1 — data and tokenizer
│   ├── download_data.py          fetch all phases from the official HF repo
│   ├── train_tokenizer.py        byte-level BPE, 32k
│   ├── eval_tokenizer.py         fertility comparison across vocab sizes
│   ├── build_memmaps.py          corpus -> flat uint16 memory maps
│   ├── profiling_data.py         corpus statistics
│   ├── measure_density.py        documents per 1024-token window, per source
│   ├── check_ordering.py         adjacent-document near-duplicate detection
│   ├── check_contamination.py    held-out set vs training overlap
│   ├── build_eval_set.py         frozen 200-example local eval set
│   ├── build_heldout.py          held-out split construction
│   └── make_clean_eval.py        contamination-filtered 179-example subset
│
├── Phase 1 — models and training
│   ├── decoder_only_transformer.py   base GPT, ModelConfig
│   ├── prefix_lm.py, prefix_gpt.py   prefix-LM
│   ├── efficient_attn.py             windowed local-global attention
│   ├── encoder_decoder.py            encoder-decoder
│   ├── span_corruption.py            T5-style corruption + long-terminal mode
│   ├── encdec_data.py                encoder-decoder batching
│   ├── doc_mask.py                   document-boundary attention masking
│   ├── sampler.py                    batch sampling from memmaps
│   ├── train.py                      training loop (all four architectures)
│   ├── local_eval.py                 BERTScore / ROUGE / BLEU local evaluation
│   ├── make_submission.py            Phase-1 submission builder
│   ├── analyse_position_loss.py      per-position loss analysis
│   └── overfit_test.py               400-step memorisation sanity check
│
├── Phase 2 — Legal NER
│   ├── profile_phase2.py  test_alignment.py  bio_tagging.py  windowing.py
│   ├── check_folds.py     ner_data.py        ner_model.py    ner_metrics.py
│   ├── train_ner.py       crf.py             lorma.py        distill_ner.py
│   ├── analyse_ner_errors.py  analyse_boundaries.py  save_ner_tags.py
│   ├── decode_variants.py     consistency_vote.py
│   └── make_ner_submission.py
│
├── Phase 3 — Rhetorical roles
│   ├── profile_phase3.py  rr_data.py  rr_model.py  rr_metrics.py
│   ├── train_rr.py        distill_rr.py  analyse_roles.py
│   ├── sweep_tau.py       logit-adjustment sweep
│   ├── eval_vote.py  vote_rr.py          ensemble over saved predictions
│   └── make_rr_submission.py
│
├── Phase 4 — Court judgment prediction
│   ├── profile_phase4.py  cjpe_data.py  cjpe_model.py
│   ├── train_cjpe.py      distill_cjpe.py
│   └── make_cjpe_submission.py
│
├── Phase 5 — Statute identification
│   ├── profile_phase5.py  lsi_data.py  lsi_model.py  lsi_metrics.py
│   ├── train_lsi.py       distill_lsi.py
│   ├── chunk_lsi.py       chunked inference + aggregation study
│   ├── ensemble_lsi.py    cross-architecture probability ensemble
│   └── make_lsi_submission.py
│
├── Phase 6 — Abstractive summarization
│   ├── profile_phase6.py  summ_data.py  summ_model.py
│   ├── summ_baseline.py   lead-k extractive control + BERTScore harness
│   ├── train_summ.py      distill_summ.py
│   └── gen_summ.py        decoding study + submission builder
│
├── Phase 7 — Prior case retrieval
│   ├── profile_phase7.py  pcr_data.py  pcr_model.py
│   ├── pcr_baseline.py    BM25 control + zero-shot dense
│   ├── train_pcr.py       distill_pcr.py
│   ├── pcr_fuse.py        reciprocal-rank fusion
│   └── make_pcr_submission.py
│
└── Tests
    ├── test_prefix_lm.py  test_prefix_docmask.py  test_efficient.py
    ├── test_encoder_decoder.py  test_encdec_generation.py  test_ner_model.py
    └── test_alignment.py  check_folds.py
```

Not in the repository (excluded by `.gitignore`): `checkpoints/` (152GB),
`training_data/` and `test_data/` (competition data, not ours to redistribute),
and generated `prediction.json` files. The `submissions/*/config.txt` files
**are** tracked — each records the checkpoint, input geometry and decision rule
behind a leaderboard score.

---

## 3. Setup

```bash
git clone https://github.com/0nibaba/CS781-Capstone.git
cd CS781-Capstone
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies (full list in `requirements.txt`):

```
torch==2.11.0+cu128    tokenizers==0.23.2    numpy==2.2.6
datasets==5.0.1        bert-score==0.3.13    rouge-score==0.1.2
nltk==3.10.3           transformers==5.17.0  matplotlib==3.10.9
```

`transformers` is required only because `bert-score` loads `roberta-large` for
evaluation. It is **not** used for any model in this project.

Download the data:

```bash
cd src
python download_data.py          # all phases, into training_data/ and test_data/
```

Expected layout:

```
training_data/raw/Phase_N_<TASK>/Train/...
test_data/Phase_N_<TASK>/Test/...
checkpoints/<run_name>/ckpt_best.pt
checkpoints/tokenizers/legal_32000.json
```

Hardware used: one NVIDIA A40 (46GB). Phase-1 runs take ~6.5 hours each at 66.3k
tokens/s (29.3% MFU); downstream runs take 15 minutes to 3 hours.

---

## 4. Phase 1 — pretraining

### 4.1 Pipeline

```bash
cd src

# 1. tokenizer (compare vocab sizes first if you want to reproduce the choice)
python train_tokenizer.py --vocab_size 32000
python eval_tokenizer.py

# 2. corpus -> memory maps  (sub-200-word docs written separately so their
#    inclusion is a runtime switch, not a re-tokenization)
python build_memmaps.py

# 3. corpus analysis that informed the design
python profiling_data.py          # token counts, mixture shares, perplexity targets
python measure_density.py         # documents per 1024-token window, per source
python check_ordering.py          # adjacent near-duplicate detection

# 4. frozen local evaluation set, then its contamination-filtered subset
python build_eval_set.py
python check_contamination.py
python make_clean_eval.py
```

### 4.2 Training the four architectures

Matched budget, 5,700 steps × 262,144 tokens = 1.494B tokens:

```bash
python train.py --run_name arm_a_baseline  --arch decoder    --seed 1899
python train.py --run_name arch2_prefix    --arch prefix_lm  --seed 1899
python train.py --run_name arch3_efficient --arch efficient  --seed 1899 --window_size 256 --global_layers 4
python train.py --run_name arch4_encdec    --arch encdec     --seed 1899
```

Document-boundary masking ablation (`--doc_mask` builds a 4D `(B,1,T,T)` mask
that blocks attention across `<|endoftext|>`):

```bash
python train.py --run_name arm_b_docmask --arch decoder --seed 1899 --doc_mask
python analyse_position_loss.py --runs arm_a_baseline arm_b_docmask
```

Longer runs for the final submissions:

```bash
python train.py --run_name arch2_prefix_3B --arch prefix_lm --seed 1899 --max_steps 11400
```

### 4.3 Evaluation and submission

```bash
python local_eval.py --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt --clean
python make_submission.py --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt \
    --temperature 0.8 --top_p 0.95 --repetition_penalty 1.1
```

**Always pass `--clean`.** Without it the evaluation set is contaminated: 42% of
High Court evaluation windows have a near-duplicate in training at Jaccard ≥ 0.5,
even after exact-MD5 deduplication removed 123,603 documents. Re-scoring on the
clean 179-example subset moved the document-masking generation gap from +0.036
to +0.0153 — roughly 60% of the original headline was memorised near-duplicates.

---

## 5. Phases 2–7 — downstream adaptation

Every phase follows the same shape: profile the data, build the representation,
train with one adaptation technique per run, then choose a decision rule. The
`--finetune` / `--use_lorma` flags select the technique; everything else is held
fixed so the comparison is clean.

### Phase 2 — Legal NER

```bash
python profile_phase2.py
python test_alignment.py             # tokenizer alignment ceiling: 98.66% macro-F1
python train_ner.py --run_name ner_crf --fold fold1 --use_crf \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python analyse_ner_errors.py --run ner_crf
python decode_variants.py --run ner_crf          # orphan-I / bridging study
python distill_ner.py --teacher_dir ../checkpoints/ner/ner_crf \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python make_ner_submission.py --sub_name p2_sub1 --run ner_crf
```

Character offsets are reconstructed under the **`trim` convention** — expand to
every token a span touches, then strip whitespace. Tested alternatives:
token-boundary alignment recovers only 3.66%; additionally stripping trailing
punctuation scores 3.6 points *worse*, because 397 gold spans legitimately end in
punctuation (217 of them citations where the bracket belongs to the entity).

The CRF hard-masks the **44% of tag-pair transitions that are structurally
impossible** rather than learning them from 56 documents. It is worth **+0.18**
cross-validated — the largest single gain in the project.

### Phase 3 — Rhetorical roles

```bash
python profile_phase3.py
python train_rr.py --run_name rr_decoder_plain --epochs 12 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python train_rr.py --run_name rr_lorma --use_lorma --max_lr 3e-4 --epochs 14 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python train_rr.py --run_name rr_ctx24 --context 24 --ctx_frac 0.5 --epochs 12 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python distill_rr.py --teacher_dir ../checkpoints/rr/rr_ctx24 \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python make_rr_submission.py --sub_name p3_sub1 \
    --rr_checkpoint ../checkpoints/rr/rr_decoder_plain/ckpt_best.pt \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
```

Sentences are separated by `<|unused0|>` and classified from that token's hidden
state; `<|unused1|>` / `<|unused2|>` mark the Corporate Law and Income Tax
domains so both train jointly. Blocks are built **predicted-region-first** with
lead-in context back-filled into the leftover budget, so no sentence is ever left
unpredicted and prediction slots always align with `[pred_from, pred_to)`.

### Phase 4 — Court judgment prediction

```bash
python profile_phase4.py
python train_cjpe.py --run_name cjpe_tail --mode tail --pooling last
python train_cjpe.py --run_name cjpe_head --mode head --pooling last      # ablation
python train_cjpe.py --run_name cjpe_lorma --finetune lorma --max_lr 3e-4 \
    --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt --pooling mean
python distill_cjpe.py --teacher_dir ../checkpoints/cjpe/cjpe_tail \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python make_cjpe_submission.py --sub_name p4_sub1 \
    --checkpoint ../checkpoints/cjpe/cjpe_tail/ckpt_best.pt \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
```

`--mode {head,tail,head_tail}` selects which 1024 tokens the model sees. This is
the largest single effect in the project: **tail 0.6997 vs head 0.5104 on the
leaderboard**. Evaluation reports macro-F1 both raw and after subtracting
`tau * log p_train(y)`, because the training prior is 59/41 while evaluation is
balanced; the submission builder applies whichever rule the checkpoint recorded.

### Phase 5 — Statute identification

```bash
python profile_phase5.py
python train_lsi.py --run_name lsi_full --mode head --pooling mean --epochs 4
python train_lsi.py --run_name lsi_topk --finetune top_k --top_k 4 --max_lr 1e-4 \
    --checkpoint ../checkpoints/arch3_efficient/ckpt_best.pt --epochs 3
python distill_lsi.py --teacher_dir ../checkpoints/lsi/lsi_full3 \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python chunk_lsi.py --checkpoint ../checkpoints/lsi/lsi_distil/distilled_best.pt \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python ensemble_lsi.py --sub_name p5_sub7 --models <ckpt>:<base> ...
```

Every evaluation reports macro-F1 three ways — fixed 0.5, best global threshold,
and per-label thresholds — because **the decision rule is worth +0.139 here
against a 0.049 spread across all four adaptation techniques**. Thresholds are
saved into the checkpoint so a submission cannot silently fall back to 0.5.

### Phase 6 — Abstractive summarization

```bash
python profile_phase6.py
python summ_baseline.py --split dev --limit 200     # lead-k extractive control
python train_summ.py --run_name summ_encdec --epochs 6 \
    --checkpoint ../checkpoints/arch4_encdec/ckpt_best.pt
python train_summ.py --run_name summ_lorma --finetune lorma --max_lr 3e-4 --epochs 6 \
    --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt
python distill_summ.py --teacher_dir ../checkpoints/summ/summ_lorma \
    --base_checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt --epochs 5
python gen_summ.py --checkpoint ../checkpoints/summ/summ_distil/distilled_best.pt \
    --base_checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt --n 50
```

`gen_summ.py` with no `--config` sweeps six decoding configurations and reports
BERTScore, mean length and 4-gram repetition rate for each. **Run the sweep
before submitting**: decoding is worth +0.040 and inverts the architecture
ranking. Under fixed greedy decoding the encoder–decoder appears to beat the
prefix-LM by 0.012; under each model's best decoding the prefix-LM wins by 0.008.

`summ_baseline.py` exists because BERTScore truncates at 512 `roberta-large`
tokens — lead-512, lead-750, lead-1000 and the *entire document* all score
exactly 0.8300. Generation fills the window and stops: EOS is suppressed until
480 tokens, hard stop at 544.

### Phase 7 — Prior case retrieval

```bash
python profile_phase7.py
python pcr_baseline.py --split dev --bm25_tokens 1024 2048 4096 8192 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python train_pcr.py --run_name pcr_full --epochs 5 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python train_pcr.py --run_name pcr_lorma --finetune lorma --max_lr 3e-4 --epochs 6 \
    --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt
python distill_pcr.py --teacher_dir ../checkpoints/pcr/pcr_full \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt --epochs 6
python pcr_fuse.py --sub_name p7_sub5 --models <ckpt>:<base> ...
```

A case is up to four 1024-token chunks over its **first 4,096 tokens**, scored by
maximum cosine similarity over chunk pairs. The budget is empirical: BM25 peaks
at 4,096 tokens (0.2991) and collapses to 0.0734 on full documents. Training
samples one chunk from each side so training and inference perform the same
operation. Chunked max-sim is worth **+0.061 with no training at all**.

---

## 6. Reproducing the best submission per phase

Each command below regenerates the `prediction.json` behind the reported score.
All submission builders read the input geometry and decision rule **from the
checkpoint**, never from the command line — see
[Known issues](#8-known-issues-and-gotchas).

```bash
cd src

# Phase 1 — 0.8343
python make_submission.py --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt \
    --temperature 0.8 --top_p 0.95 --repetition_penalty 1.1

# Phase 2 — 0.5683
python train_ner.py --run_name ner_crf_all --use_crf --all_data --epochs 8 \
    --checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt
python make_ner_submission.py --sub_name p2_best --run ner_crf_all

# Phase 3 — 0.5023
python make_rr_submission.py --sub_name p3_best \
    --rr_checkpoint ../checkpoints/rr/rr_decoder_plain/ckpt_best.pt \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt

# Phase 4 — 0.6997
python make_cjpe_submission.py --sub_name p4_best \
    --checkpoint ../checkpoints/cjpe/cjpe_tail/ckpt_best.pt \
    --base_checkpoint ../checkpoints/arm_b_docmask/ckpt_best.pt

# Phase 5 — 0.3978
python ensemble_lsi.py --sub_name p5_best \
    --models ../checkpoints/lsi/lsi_distil/distilled_best.pt:../checkpoints/arm_b_docmask/ckpt_best.pt \
             ../checkpoints/lsi/lsi_full3/ckpt_best.pt:../checkpoints/arm_b_docmask/ckpt_best.pt \
             ../checkpoints/lsi/lsi_topk/ckpt_best.pt:../checkpoints/arch3_efficient/ckpt_best.pt \
             ../checkpoints/lsi/lsi_lorma/ckpt_best.pt:../checkpoints/arch2_prefix_3B/ckpt_best.pt \
             ../checkpoints/lsi/lsi_encdec/ckpt_best.pt:../checkpoints/arch4_encdec/ckpt_best.pt

# Phase 6 — 0.8260
python gen_summ.py --sub_name p6_best --config greedy-480-norep4 \
    --checkpoint ../checkpoints/summ/summ_distil/distilled_best.pt \
    --base_checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt

# Phase 7 — 0.2965
python pcr_fuse.py --sub_name p7_best \
    --models ../checkpoints/pcr/pcr_full/ckpt_best.pt:../checkpoints/arm_b_docmask/ckpt_best.pt \
             ../checkpoints/pcr/pcr_lorma/ckpt_best.pt:../checkpoints/arch2_prefix_3B/ckpt_best.pt \
             ../checkpoints/pcr/pcr_topk/ckpt_best.pt:../checkpoints/arch3_efficient/ckpt_best.pt \
             ../checkpoints/pcr/pcr_distil/distilled_best.pt:../checkpoints/arm_b_docmask/ckpt_best.pt
```

Each writes `submissions/<sub_name>/` containing `prediction.json`,
`submission.zip` and a `config.txt` recording the exact checkpoint, geometry and
decision rule used.

---

## 7. Verification suite

These were written and run **before** committing GPU time, not afterwards:

```bash
cd src
python test_prefix_lm.py           # prefix-LM masking and loss positions
python test_prefix_docmask.py      # prefix-LM + document masking interaction
python test_efficient.py           # windowed attention mask correctness
python test_encoder_decoder.py     # cross-attention, causality in the decoder
python test_encdec_generation.py   # generation reaches the required length
python test_ner_model.py           # CRF forward/Viterbi, head shapes
python test_alignment.py           # character-offset round-trip ceiling
python overfit_test.py             # 400-step memorisation to loss 0.0000
```

What they check: manual and FlashAttention paths agree to 1.5e-7; perturbing
position 10 leaves positions 0–9 bit-identical (causality); perturbing padded
positions leaves real positions bit-identical; zeroed residual projections make
output equal input; initial loss is 10.44 against ln(32000) = 10.37; and a
400-step overfit reaches 0.0000 with verbatim reconstruction including section
references such as "173(2)".

---

## 8. Known issues and gotchas

**Input geometry must match between training and inference.** A Phase-3
submission built test blocks with default context settings against a checkpoint
trained with different geometry and lost 0.0095 on the leaderboard (0.5014 →
0.4919). Every submission builder now reads `context`, `ctx_frac`, `max_len`,
`mode`, `pooling` and the decision rule from `ck["args"]` and prints them. The
symptom was visible beforehand: the majority label occupied 43.7% of test
sentences against 32% on development for the same model. **Compare the submitted
label distribution against the model's own development distribution before
uploading.**

**LoRMA and distilled checkpoints need reconstruction before loading.**
A LoRMA checkpoint has wrapped `c_attn` modules, so `apply_lorma()` must run
*before* `load_state_dict`. A distilled student has half the layers, so the
backbone must be truncated using the saved `keep_every` first. All submission
builders handle both; if you write a new one, copy that block.

**Adapter creation order matters.** `apply_lorma()` must run after the model
constructor (so the walk finds `c_attn`), before `.to(device)` (adapters are
created on CPU), and before the optimizer is built (it freezes parameters).
Getting this wrong produces a device-mismatch crash or silent full fine-tuning.

**`--all_data` deletes early stopping.** Retraining on train+dev before
submitting cost 0.040 on the Phase-3 leaderboard. With a dataset a 110M model can
memorise, the held-out split is worth more as a stopping signal than as 12.5%
extra training data. Only use `--all_data` with an epoch count validated
beforehand, and preferably not at all.

**BitFit does not work on this architecture.** The models have no bias terms, so
a BitFit arm trains only the classification head. The arm is retained as a
head-only probe because it is informative, but it is not BitFit.

**Phase-3 development set is too small to rank models.** Ten documents; within a
single run, epoch-to-epoch scores have a standard deviation of 0.021 while
training loss falls monotonically. Best-of-twelve-epoch selection inflates by
roughly 0.03 and varies per run. Treat every Phase-3 comparison as weak evidence.

**Network restrictions on the training machine.** Campus TLS interception breaks
`pypi` and `huggingface.co`, so `datasketch` could not be installed and the
near-duplicate check in `check_ordering.py` is an adjacency search rather than
MinHash. `bert-score` works only because `roberta-large` is already in the local
HuggingFace cache.
