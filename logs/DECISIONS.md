# Phase 1 decision log

## 2026-09-25 — corpus profiling
3.07B words / ~4.0B tokens across four sources.
Natural mixture: high 54.9%, tribunal 34.4%, supreme 6.8%, acts 3.9%.
Median doc lengths: high 282w, tribunal 628w, supreme 2378w, acts 2624w.
34.6% of High Courts docs are <200 words.
118k exact duplicates in high, 4.9k in tribunal, none cross-source.
Test set: 1000 prompts, all exactly 512 words.

**Implications:** natural mixture over-represents short High Court stubs
relative to the test distribution (which includes statutory text).
Median doc length < context length, so document separators are necessary
and long context has weak justification.

## 2026-09-25 — tokenizer
Trained BPE at 16/24/32/48k, with and without explicit digit splitting.
Sample: 350MB, reweighted toward acts+supreme (100/150/50/50 MB).
Held-out eval: 300 docs/source, disjoint by byte offset.

| tokenizer | overall fertility | emb @768 |
|---|---|---|
| gpt2 | 1.353 | 38.6M |
| legal_24000 | 1.314 | 18.4M |
| legal_32000 | 1.296 | 24.6M |
| legal_48000 | 1.279 | 36.9M |
| legal_ds_32000 | 1.394 | 24.6M |

**Decisions:**
- Vocab 32k (or 24k). 48k buys 1.3% fertility for 12.3M params — bad trade.
- No explicit digit splitting: costs 7.6% fertility, and byte-level BPE
  already splits digits without it (verified in token dump).
- Corpus is ~4.0B tokens at 32k vocab.

**Open:** 24k vs 32k — the 6.2M param difference is roughly one extra
layer at d_model=768. Worth testing empirically.

The two matrices

Input embedding, shape (vocab_size, d_model) = (32000, 768).

Row i is the vector for token i. When token 4271 comes in, you fetch row 4271. So a row here means: "when this token appears, here is what it contributes to the model's representation."

Output projection, shape (d_model, vocab_size) = (768, 32000).

At the end, you have a 768-dim vector representing "everything the model knows about what comes next." You multiply by this matrix to get 32,000 scores, one per vocabulary entry. Column j is dotted against that vector to produce token j's score. So a column here means: "here is the pattern that, if present in the final vector, means this token should come next."

Why tying them works

Both matrices are (32000, 768) if you transpose one. Tying means using the same parameters for both:

logits = h @ E.T        where E is the embedding table

The output score for token j becomes the dot product between the final hidden state and token j's embedding vector.

Read that again, because it's the whole idea: the model predicts a token by producing a hidden state that points in the same direction as that token's embedding. Prediction becomes a nearest-neighbour search in embedding space.

That's a coherent thing to ask of the geometry. The word that goes in and the word that comes out are the same word, so they should live at the same point. If "petitioner" is represented one way on input and a completely unrelated way on output, you're learning the same semantic fact twice.

And the empirical evidence agrees: untied models tend to learn output columns that end up similar to their input rows anyway. Tying just tells the model that up front instead of making it discover it.

What it costs and saves

At 32k × 768, the table is 24.6M parameters. Untied, you pay it twice: 49.2M. Tied, once.

Against a 150M budget that's 24.6M freed — enough for roughly two extra transformer layers at d_model=768 (each layer is ~12 × 768² ≈ 7M).

There's a regularisation benefit too: fewer parameters, and the embedding gets gradient signal from both ends of the network.

The tension, which is real

The two roles aren't identical. Input embeddings mostly want to encode "what this token means." Output columns want to encode "what contexts predict this token." Related, not the same.

Frequency behaves differently at each end too — common tokens dominate the output softmax in a way they don't dominate input lookups.

Very large models sometimes untie for this reason. At your scale, with a fixed parameter budget, tying is near-universally the right call: the parameters buy you more as depth than as a second embedding table.

## 2026-09-25 — held-out set construction

Built a held-out pool disjoint from tokenizer training data by byte offset
(skipped past the first 150/200/80/80 MB of acts/supreme/high/tribunal,
exceeding the 100/150/50/50 MB the tokenizer sampler consumed).

- eval_sample.json: 300 docs/source, ≥300 words, for tokenizer fertility
- heldout_pool.json: 500 docs/source, ≥700 words, for 512/128 generation eval
- heldout_hashes.json: MD5 of raw text, used to exclude from training memmaps

**Verified:** build_memmaps skipped 800 held-out docs per source, matching
300+500. Separation confirmed by assertion, not assumed.

## 2026-09-25 — weight tying

Input embedding and output projection share one (32000, 768) matrix.
Rationale: prediction becomes a dot product between the final hidden state
and each token's embedding, i.e. nearest-neighbour in embedding space —
the same word going in and coming out should occupy the same point.
Saves 24.6M params (~2 extra layers at d_model=768).

**Check:** model.lm_head.weight is model.token_emb.weight must be True;
parameter count must drop by exactly 24,576,000.
**Expected initial loss:** ln(32000) ≈ 10.4. Deviation indicates an
initialization problem.

## 2026-09-25 — memmap pipeline

Tokenized to flat uint16 .bin files, one per source, with <|endoftext|>
appended after each document. Separate files for sub-200-word documents
so short-doc inclusion is a runtime switch rather than a re-tokenization.

Document separators are necessary here: median High Courts doc is 282 words
vs a 1024–2048 token block, so blocks pack 5–6 unrelated cases.

Dedup by MD5 removed 118,437 High Courts / 4,934 Tribunal docs — matches
the profiling counts exactly.

Token counts (main / short):
- high: 1,497,303,564 / 175,743,682
- [fill in acts, supreme, tribunal]

**Note:** short docs are 34.6% of High Courts by document count but only
~10.5% by token count. Excluding them costs less than expected.

## 2026-09-25 — local evaluation

Built a fixed 200-example eval set: random mid-document 512-word windows
with the following 128 words as reference, stratified across sources,
seed 42, no sentence-boundary snapping.

Rationale: the real test prompts are arbitrary mid-document slices ending
mid-sentence (verified: "...on grounds of laches. It is"). A tidier eval
set with clean sentence breaks would be a different distribution.

Scoring: BERTScore F1 (rescale_with_baseline=False to match the likely
Codabench default), plus ROUGE-L and BLEU-4. Diagnostics logged per
generation: word count, distinct-4-gram ratio (repetition detector),
whether output fell short of 128 words.

**Open:** BERTScore default is roberta-large layer 17. If Codabench uses
a different config, absolute numbers won't match the leaderboard even
though rankings will. Check the published evaluation script.

## 2026-09-25 — memmap pipeline results (verified)

Tokenizer: legal_32000 (BPE, byte-level, no explicit digit split).
<|endoftext|> = token id 0, appended after every document.
Documents below 200 words routed to separate *_short.bin files so their
inclusion is a runtime switch, not a re-tokenization.

Final token counts (main / short):

| source   | main tokens   | short tokens | share of main |
|----------|---------------|--------------|---------------|
| tribunal | 1,616,880,704 |   22,710,879 | 46.1%         |
| high     | 1,497,303,564 |  175,743,682 | 42.7%         |
| supreme  |   252,591,591 |      299,078 |  7.2%         |
| acts     |   141,839,507 |       77,900 |  4.0%         |
| **total**| **3,508,615,366** | **198,831,539** |         |

Removed 123,603 exact duplicates (matches profiling counts exactly).
Skipped 3,222 held-out documents — assertion passed, separation verified.

**Verified on disk:** all four .bin files decode to clean legal prose,
lengths match meta.json, max token id = 31999 < 32000.

**Mixture shifted vs raw profiling.** Tribunal is now the largest source
by tokens (46.1%), overtaking High Courts, because High lost 118k duplicates
and 931k documents to the short bucket. Acts + Supreme together are only
11.2% of the corpus.

**Constraint on upweighting:** Acts is only 142M tokens total. Reaching
20% of a 3B-token run would require ~4.2 epochs over Acts, risking
memorization of that source. Mixture ratio needs to be an experiment,
not an assumption — the ceiling on upweighting is set by repetition risk,
not by preference.

**Caution:** eot_id = 0. A zeroed or uninitialized array would look like
document separators rather than erroring. Keep in mind when debugging.

## 2026-09-25 — boilerplate observation

Decoded samples confirm each source opens with structural boilerplate:
- supreme: "PETITIONER: ... Vs. ... RESPONDENT: ... DATE OF JUDGMENT: ...
  ACT: ... HEADNOTE: ..."
- high: "Court No. - 35 Case :- WRIT - A No. - 14738 of 1989 Petitioner :- ...
  Counsel for Petitioner :- ..."
- tribunal: citation line then "JUDGMENT"

Test prompts are mid-document prose, never headers. So some fraction of
training signal teaches output the model is never scored on.
--strip_headers flag exists but is OFF for this build; worth testing as a
controlled comparison rather than assuming clean is better — the test data
itself contains OCR noise (stray page numbers), so aggressive cleaning
risks a train/test mismatch.

Also noted: supreme documents contain a HEADNOTE field (human-written
summary) inline in the text. Relevant to Phase 6.

## 2026-09-25 — model sizing

3.51B main tokens (3.71B with short docs).
At the Chinchilla-optimal ~20 tokens/parameter this supports a
**~175M parameter model** — comfortably inside the 50–300M rule.

For contrast the starter config was 59M params on 200M tokens
= 3.4 tokens/param, roughly 6x undertrained.

## 2026-09-25 — own generation loop

Decided against HuggingFace generate()/from_pretrained().

Reasons:
1. The four Phase-1 architectures are custom; HF's generate() only works
   on PreTrainedModel subclasses, so using it would mean writing wrapper
   boilerplate for each architecture including the encoder-decoder.
2. Sampling (temperature, top-k, top-p, repetition penalty) is explicitly
   part of what Phase 1 is meant to teach, and is likely viva material.
3. Full control over EOS suppression and left-padded batching, which is
   where the starter code has a silent bug.

Known cost: no KV cache in the first version, so generation is O(n²) per
sequence. Acceptable for a 200-example local eval; should be added before
running 1000-prompt Codabench generations.

## 2026-09-25 — decoder-only architecture verified

110,316,288 params (85.0M transformer, 24.6M tied token emb, 0.8M pos emb).
Embeddings are 23% of the model vs 47% in the starter config.
At 3.51B tokens this is 31.8 tokens/param — past Chinchilla-optimal (20:1),
so there is headroom to scale toward ~175M if throughput allows.

Design choices: pre-norm, RMSNorm (fp32 internally), GELU, learned absolute
positions, weight-tied, bias=False, residual projections scaled by
1/sqrt(2*n_layer).

Verified:
- manual and FlashAttention paths agree to 1.5e-07
- causality exact (perturbing position 10 leaves 0-9 bit-identical)
- padding mask exact (perturbing pads leaves real positions bit-identical)
- residual path clean (zeroed projections => output == input)
- tying saves exactly 24,576,000 params
- initial loss 10.4432 vs ln(32000) = 10.3735
- overfit test: 10.38 -> 0.0000 over 400 steps, verbatim reconstruction
  including "173(2)" and "156(3)"

## 2026-09-26 — training loop verified

Custom loop (not HF Trainer). Verified before launching anything long:

- **Sampler:** targets shifted by exactly one (X[:,1:] == Y[:,:-1]), decodes
  to readable legal text, max token id 31703 < 32000, document separators
  appear 0–3 per 1024-token window varying by source — consistent with the
  measured document-length distribution.
- **Validation split:** last 5M tokens of each .bin, never drawn by the
  training sampler. Fixed RNG seed (1337) so the same windows are scored
  at every eval — val curve reflects the model, not sampling noise.
  This is separate from eval_set.json, which measures generation.
- **Resume:** killed at step 50, resumed, val loss came back 6.7106 vs
  6.7106 saved. Model and optimizer state restored exactly.

Gotcha found: PyTorch 2.6+ defaults torch.load(weights_only=True), which
rejects the numpy RNG state in the checkpoint. Needs weights_only=False.

## 2026-09-26 — throughput measurement

A40, bf16 autocast, TF32 matmul enabled, FlashAttention via SDPA.

| batch | accum | tok/s  | MFU   |
|-------|-------|--------|-------|
| 8     | 4     | 61,000 | 27.0% |
| 32    | 8     | 66,500 | 29.4% |

Only a 9% gain from 4x the micro-batch, so the loop is closer to
compute-bound than data-bound. Memory at batch 32: 29.1 GB of 46 GB,
GPU util 100%, 290W/300W, 74°C.

Wall-clock at 66.5k tok/s: 1B ≈ 4.2h, 2B ≈ 8.4h, 3.5B ≈ 14.6h.

**Untested:** torch.compile (expected 1.3–1.5x) and batch 64.
Worth measuring before the long scaled run.

## 2026-09-26 — first training run

Decided against going straight to 3.5B. Reasons:

1. The four-architecture comparison is the graded requirement; a single
   long run is not. Architecture rankings converge early, so 1.5B is
   enough to rank them.
2. No hyperparameter is tuned yet (LR 6e-4, natural mixture, block 1024).
   Committing 14.6h to an untuned first configuration risks discovering
   a bad LR after a night of GPU time.
3. Training is serial on one GPU — a 14.6h run blocks building
   architectures 2–4.

Launched: decoder_base, 5700 steps × 262,144 tok = 1.49B tokens, ~6.5h.
LR 6e-4 peak / 6e-5 min, 1000 warmup, cosine decay, AdamW (0.9, 0.95),
weight decay 0.1 on 2D params only, grad clip 1.0, natural mixture
(acts 4.0 / high 42.7 / supreme 7.2 / tribunal 46.1).

Checkpoints every 500 steps → 11 snapshots for the val-loss vs BERTScore
comparison the brief asks for.

**Confound to note:** cosine decay is tied to max_steps, so a 5700-step
run is not a prefix of a 13400-step run. Fine for comparing architectures
at identical max_steps; must be stated if comparing across budgets.

Smoke-test behaviour (50 steps, 1.6M tokens): loss 10.54 → 6.38,
grad norm 18 → <1, all four per-source val losses tracking down together.
Nothing pathological.

## 2026-09-26 — decoder_base training results

5700 steps, 1.494B tokens, 6.50 hours, 66.3k tok/s steady throughout
(no thermal throttle, no contention). MFU 29.3% flat across the run.

Val loss 10.578 → 1.7149 (perplexity 5.56).

**Still improving at the end** — val loss fell from 1.7180 to 1.7149 on the
final eval. The cosine schedule ended the run, not the data. More tokens
would help.

Per-source validation loss:

| source   | val loss | perplexity | share of mixture |
|----------|----------|------------|------------------|
| tribunal | 1.497    | 4.47       | 46.1%            |
| high     | 1.790    | 5.99       | 42.7%            |
| acts     | 2.109    | 8.24       | 4.0%             |
| supreme  | 2.446    | 11.5       | 7.2%             |

**Key finding:** the two sources the model handles worst (supreme, acts)
are the two barely represented in the mixture, and they are the two the
test set emphasises. 88.8% of training compute goes to the two easiest
distributions. This is now a quantified argument for the mixture
experiment rather than an intuition.

**Caveat on perplexity 5.56:** low even for legal text. Two benign reasons
(legal prose is formulaic; the val split is the tail of the same .bin files,
so same distribution rather than a genuinely held-out one). Worth checking
what the val tail region actually contains — if the source JSONs were
ordered by date or court, the tail may be systematically unlike the body.

## 2026-09-26 — first local generation eval

ckpt_best.pt (step 5700) on the 200-example eval set,
T=0.8, top_p=0.95, rep_penalty=1.1, EOS suppressed, seed 0.

BERTScore F1 **0.8294** (P 0.8270 / R 0.8320)
ROUGE-L 0.2020, BLEU-4 0.0434

Diagnostics:
- mean words 127.9 / 128 target — length control works
- distinct-4gram 0.9941 — essentially no repetition, no degenerate loops
- 2.5% of generations short of 128 words — EOS suppression working

Per-source BERTScore: acts 0.8282, high 0.8305, supreme 0.8286,
tribunal 0.8304.

**Second key finding — loss and generation quality diverge.** Validation
loss varies enormously by source (supreme 2.45 vs tribunal 1.50, a 2.6x
perplexity gap) while BERTScore is nearly flat across sources (0.8282–0.8305,
a 0.3% spread). Perplexity says Supreme Court is much harder to model;
BERTScore says it generates about as well as anything else. This is direct
evidence for the relationship the brief asks about between validation
loss/perplexity and generation quality — they are not measuring the same
thing.

ROUGE-L and BLEU-4 are low in absolute terms, as expected: exact n-gram
overlap with an arbitrary 128-word continuation is near-impossible.

**Known issue:** 10.7 min for 200 examples (3.2 s/generation) because there
is no KV cache — each of the 220 generation steps re-runs the full forward
pass over the whole prompt. Extrapolates to ~54 min per 1000-prompt
Codabench submission. Fix before running the decoding sweep.

## 2026-09-26 — first Codabench submission

submissions/submission_1/ — ckpt_best.pt (step 5700), decoding T=0.8,
top_p=0.95, rep_penalty=1.1, EOS suppressed, seed 0. Same config as the
local eval, so the two numbers are directly comparable.

Validated before upload: all 1000 test IDs present, none extra, no
generation over 128 words, archive contains exactly prediction.json at
the root.

Prompt lengths measured against the truncation limit (block_size 1024 −
max_new_tokens 220 = 804): [fill in min/median/max] — no prompts
left-truncated.

Generation took ~30 min at batch 32 for 1000 prompts (no KV cache).

**Leaderboard score: [fill in]** vs local 0.8294.
If close, the local evaluator is trustworthy and tuning can proceed
against it. If not, check the competition's evaluation script for a
different BERTScore model/layer.

## 2026-09-26 — reframing: what the capstone actually rewards

Course instructor clarified: the grading is not on leaderboard position
but on **algorithmic innovation under compute and size constraints**. A
standard architecture with a good score is explicitly described as "not
that interesting"; a novel architecture or training method is highly
valued even if it scores lower.

**Honest assessment of where we stand.** The current decoder is
competent but standard: pre-norm, RMSNorm, GELU, learned absolute
positions, tied weights, SDPA/FlashAttention. Every choice is
defensible and measured, but none is original. Under this criterion the
model itself is not yet a contribution.

**What does count so far, and is already evidenced:**

1. *Loss and generation quality measure different things.* Per-source
   validation perplexity varies 2.6x (supreme 11.5, tribunal 4.47) while
   per-source BERTScore is flat within 0.3% (0.8282–0.8305). Directly
   answers a question the brief poses.

2. *The mixture problem is quantified and bounded.* The two worst-modelled
   sources (supreme 7.2%, acts 4.0%) are the two the test set emphasises.
   But Acts holds only 142M tokens, so reaching 20% of a 3B-token run
   requires ~4.2 epochs over it — upweighting is capped by memorisation
   risk, not by preference. This is a real constraint, not a
   hyperparameter.

3. *Digit splitting: hypothesis, experiment, conclusion.* Predicted it was
   redundant; measured 7.6% fertility cost; token dump confirmed
   byte-level BPE already splits digits without the extra pre-tokenizer.

**Candidate directions for a genuine contribution**, each motivated by a
measured property of this corpus rather than picked off a list:

- **Document-boundary-aware attention.** Median High Courts doc is 282
  words vs a 1024-token block, so blocks pack 5–6 unrelated cases and
  standard causal attention learns transitions between unrelated
  judgments. Masking attention at <|endoftext|> boundaries so each
  document attends only within itself is concrete, implementable, and
  ablatable against the existing baseline.
- **Source curriculum.** Per-source val loss is already logged every 250
  steps, so there is signal to drive easy→hard ordering, the reverse, or
  dynamic reweighting.
- **Per-source loss weighting** rather than data reweighting — upweight
  the gradient contribution of acts/supreme without seeing those tokens
  more often, sidestepping the repetition ceiling identified above.
  Untested; that is what makes it worth testing.

**Plan:** pick one, implement it, ablate against decoder_base
(val 1.7149, BERTScore 0.8294) at matched tokens and matched parameters,
and document the failure modes. A negative result that is properly
measured and explained is still a result.

## 2026-09-26 — experiment 1: document-boundary-aware attention

**PRE-REGISTERED before running. Predictions recorded in advance so a
null result cannot be rationalised afterwards.**

### Honest positioning

Intra-document attention masking is NOT novel. It is standard in
large-scale pretraining (T5, Llama 3) and studied directly by Zhao et al.
(2024) on sequence composition. Claiming invention would be false and
would not survive viva.

The defensible claim is about **regime**, not mechanism. The literature
studies this on web text, where documents are long and boundary density
is low. This corpus is an extreme case: median High Courts document is
282 words against a 1024-token block, and the sampler check measured
0–3 <|endoftext|> separators per window. Cross-document contamination
should therefore be far more severe here than anywhere it has been
characterised. Known method, under-studied regime, careful measurement.

### Mechanism

Per-position document index = cumulative count of EOS tokens.
Attention mask becomes `causal & (doc_id[q] == doc_id[k])`.

Convention to fix and state: the EOS token belongs to the document it
terminates (not the one it precedes). This decides whether a document's
final token may attend to its own terminator.

### Predictions (registered in advance)

**P1 — differential by source.** Benefit must concentrate where boundary
density is high. Expected ordering of improvement:
  high (median 282w) > tribunal (628w) >> supreme (2378w) ≈ acts (2624w)
If aggregate loss improves but the per-source pattern is flat, the
mechanism is NOT what is claimed and this must be reported as such.

**P2 — position within block.** Contamination should worsen with
accumulated unrelated context, so the gain should grow at later
positions within a block.

### Confounds to control

- **Early-document positions.** Under masking, the first token of
  document 2 has almost no context; previously it had garbage context.
  Aggregate loss could worsen even if the model is genuinely better.
  Loss must be conditioned on position-within-document, not only
  position-within-block.
- **Position embeddings.** Learned absolute positions mean document 2
  currently sits at positions ~400–700. Resetting positions per document
  is arguable but is a SECOND change — run as a separate arm or the
  effect cannot be attributed.
- **Matching.** Identical seed, data order, tokens, steps, parameters.
  Only the mask differs.

### Engineering note

Passing an explicit attn_mask disables SDPA's `is_causal` fast path and
loses the FlashAttention kernel. Baseline throughput is 66.5k tok/s;
expect a significant drop. PyTorch 2.11 has FlexAttention, which
expresses this as a `mask_mod` and compiles to a block-sparse kernel.
**Measure both naive and FlexAttention throughput** — the comparison
demonstrates why the fast path exists and is a report section in itself.

### Verification (same pattern as the existing causality/padding tests)

- Unit test mask construction against a hand-computed small example.
- Build a block with known EOS positions, perturb a token in document 1,
  assert every position in document 2 is bit-identical. This is the
  direct analogue of the checks that already pass for causality and
  padding.

### Ablation plan

Baseline to beat: decoder_base, 1.494B tokens, val 1.7149,
per-source val [tribunal 1.497 / high 1.790 / acts 2.109 / supreme 2.446],
BERTScore 0.8294.

Arms, each at matched tokens and matched parameters (~6.5h each):
  A. baseline (done)
  B. + document-boundary masking
  C. + masking + per-document position reset (only if B is informative)

Possible extension toward something more original: a learned per-head
gate on cross-document attention, letting the model decide how much to
let through rather than hard-masking. Untested.

**A negative result that was predicted in advance and explained is a
valid contribution. A positive result rationalised after the fact is
not.**

## 2026-09-27 — arm A complete, seed variance measured

arm_a_baseline (seed 1899, no mask) finished: 5700 steps, 1.494B tokens.
Final val 1.72031. Per-source: tribunal 1.505 / high 1.794 / acts 2.108 /
supreme 2.450.

**Seed variance, measured.** decoder_base (unseeded, effectively a
different seed) reached val 1.7149 on identical settings. arm_a reached
1.72031. **Δ = 0.005.**

Per-source agreement is similarly tight:
| source   | decoder_base | arm_a (1899) | Δ     |
|----------|--------------|--------------|-------|
| tribunal | 1.497        | 1.505        | 0.008 |
| high     | 1.790        | 1.794        | 0.004 |
| acts     | 2.109        | 2.108        | 0.001 |
| supreme  | 2.446        | 2.450        | 0.004 |

This is the calibration the whole ablation rests on: **differences below
~0.005–0.008 cannot be distinguished from seed noise.** Anything larger
is likely real. This was obtained free, as a byproduct of retraining the
baseline under a fixed seed, and turns "n=1 per arm" from a bare
limitation into a quantified one.

## 2026-09-27 — arm B interim (step 3250 of 5700)

GPU was surrendered mid-run to another user with a deadline. arm B
resumes from ckpt_latest at step 3250.

**Document masking is currently WORSE, and the gap is stable:**

| step | arm A (no mask) | arm B (doc mask) | Δ (B − A) |
|------|-----------------|------------------|-----------|
| 3000 | 1.87151         | 1.89350          | **+0.0220** |
| 3250 | 1.84453         | 1.86670          | **+0.0222** |

+0.022 is roughly 4x the measured seed variance, so this is very likely a
real effect rather than noise.

Per-source at step 3250:
| source   | arm A   | arm B   | Δ      |
|----------|---------|---------|--------|
| tribunal | 1.6184  | 1.6642  | **+0.046** |
| acts     | 2.2802  | 2.2914  | +0.011 |
| high     | 1.9214  | 1.9227  | +0.001 |
| supreme  | 2.5945  | 2.5951  | +0.001 |

**This contradicts pre-registered prediction P1, and in an informative
way.** P1 predicted the *benefit* would concentrate where boundary
density is highest (high > tribunal >> supreme ≈ acts). Instead the
*penalty* concentrates on tribunal, while `high` — the source with the
shortest documents and therefore the most boundaries per block — is
essentially unaffected. The predicted ordering does not appear at all.

**Two candidate explanations. Neither is concluded yet; the per-position
instrumentation is what distinguishes them.**

1. *The early-document confound, which was pre-registered.* Under
   masking, positions near the start of each packed document lose the
   (garbage) context they previously had. With several documents per
   block there are many such positions, so aggregate loss rises even if
   the mechanism is sound. The pos_loss_*.npz arrays will show whether
   the penalty is concentrated at low position-within-document.

2. *Cross-document context may genuinely help here.* Unrelated Indian
   legal documents still share register, vocabulary, citation format and
   formulaic phrasing. Attending across a boundary may supply useful
   domain priming rather than noise. If the position analysis does not
   support explanation 1, this becomes the more likely reading — and a
   more interesting result than "masking helps", since it would suggest
   the standard practice is not universally beneficial and that the
   legal-domain regime differs from web text.

**Not yet concluded.** Arm B has 2450 steps remaining and early-training
gaps can close. Recording the interim result and the P1 contradiction
here, before the run completes, so the analysis cannot be retrofitted to
whatever the final number turns out to be.

**Throughput note.** Arm B ran at 28–31k tok/s vs arm A's 65k, but this
is mostly GPU contention from another user rather than the masking cost
(measured separately at 19.4% end-to-end). Wall-clock figures across the
two arms are therefore not comparable; token-matched loss is unaffected.

Both arms were interrupted and resumed at least once. RNG state is
restored on resume, but the interruption points differ between arms, so
the data sequences are not bit-identical to what uninterrupted runs
would have produced.

## 2026-09-27 — leaderboard vs local: evaluator validated

Submission 1 (decoder_base ckpt_best, T=0.8 p=0.95 rp=1.1, seed 0):

| metric        | local (200) | Codabench (1000) | gap     |
|---------------|-------------|------------------|---------|
| BERTScore F1  | 0.8294      | **0.82601**      | −0.0034 |
| ROUGE-L F1    | 0.2020      | 0.16976          | −0.032  |
| BLEU-4        | 0.0434      | 0.02754          | −0.016  |

**BERTScore agrees to 0.34%.** The local evaluator is a reliable proxy
for the leaderboard, so decoding and checkpoint selection can be tuned
locally without spending Codabench submissions (30 min each, 10/day cap).
This was the purpose of submitting early and it is confirmed.

Secondary metrics diverge more in relative terms, as expected: exact
n-gram overlap against an arbitrary 128-word continuation is inherently
high-variance, and they are secondary.

Note the local set is 200 examples vs 1000; the small negative offset is
consistent with sampling and with a slightly different private test
distribution.

## 2026-09-27 — separator density measured (the independent variable)

2000 random 1024-token windows per source, training region only, seed 1899.

| source   | mean docs/win | median | p90 | max | % single-doc | mean fragment |
|----------|---------------|--------|-----|-----|--------------|---------------|
| high     | 1.99          | 2      | 3   | 5   | 38.0%        | 515 tok       |
| tribunal | 1.49          | 1      | 2   | 4   | 60.3%        | 688 tok       |
| supreme  | 1.22          | 1      | 2   | 4   | 80.1%        | 842 tok       |
| acts     | 1.12          | 1      | 2   | 3   | 88.8%        | 916 tok       |

**The original motivation was overstated, and this corrects it.** The
experiment was premised on the raw document-length distribution (median
High Courts doc = 282 words), implying 5–6 documents per block. The
actual figure is 1.1–2.0, because sub-200-word documents were routed to
*_short.bin and are NOT in the training mixture. What remains is
substantially longer. Even `high` is single-document 38% of the time.
Cross-document contamination is therefore a much milder phenomenon here
than assumed when the experiment was designed. Stated plainly rather
than quietly dropped.

Density ranking: high (1.99) > tribunal (1.49) > supreme (1.22) > acts (1.12).

## 2026-09-27 — position analysis: BOTH pre-registered predictions refuted

Analysis at step 3250 (last step matched across both arms).
Positive delta = masking is worse.

|          | arm A  | arm B  | delta   | first 64 | last 64 | slope   | density |
|----------|--------|--------|---------|----------|---------|---------|---------|
| overall  | 1.8445 | 1.8667 | +0.0222 | −0.0110  | +0.0496 | +0.0666 |         |
| tribunal | 1.6184 | 1.6642 | +0.0458 | −0.0218  | +0.1007 | +0.1384 | 1.49    |
| acts     | 2.2802 | 2.2914 | +0.0112 | +0.0203  | +0.0164 | +0.0084 | 1.12    |
| high     | 1.9214 | 1.9227 | +0.0013 | −0.0045  | +0.0061 | +0.0061 | 1.99    |
| supreme  | 2.5945 | 2.5951 | +0.0007 | +0.0021  | −0.0015 | −0.0020 | 1.22    |

**P1 refuted.** Predicted: benefit concentrates where boundary density is
highest (high > tribunal >> supreme ≈ acts). Observed: the *penalty*
ranking is nearly the reverse of the density ranking. `high` has twice
the density of `acts` and shows 1/9th the effect. `tribunal` has 25%
fewer boundaries than `high` and shows 35x the penalty. **The effect does
not track boundary density across sources.**

**P2 refuted, with the sign reversed.** Predicted: benefit grows at later
positions as contamination accumulates. Observed: the *penalty* grows at
later positions (slope +0.067 overall, +0.138 for tribunal). Masking is
slightly *better* at early positions (−0.011 overall) and clearly worse
by the end of the block (+0.050).

**The pre-registered early-document confound is ruled out.** It predicted
masking would hurt most where context is removed, i.e. early positions.
The data shows the opposite — early positions are exactly where masking
helps. The penalty is not a boundary artifact.

## 2026-09-27 — two findings that emerged from the figures

**1. The effect switches on at position ~330, where the first boundary
typically falls.** In docmask_delta_by_position.png both the overall and
tribunal curves sit flat at zero from position 0 to roughly 330, then
rise sharply. Mean fragment length is 515–916 tokens, so positions
300–700 is where a typical window crosses its first document boundary.
Before that point most windows contain no boundary, masking is a no-op,
and the two arms are numerically identical — which is what the flat
region shows.

This is a strong internal consistency check: the effect turns on exactly
where the measured density says it must. Whatever the sign of the result,
the mechanism is demonstrably doing what it is supposed to.

**2. The sign of the effect flips during training**
(docmask_delta_over_training.png). The penalty is strongly *negative*
early — −0.118 at step 250, i.e. masking is much better — crosses zero
near step 1000, and stabilises at +0.022 from step 2000 onward.

Interpretation, offered as a hypothesis rather than a conclusion:
cross-document context is unusable noise to an undertrained model, but
becomes usable signal once the model has learned legal register.
Unrelated Indian legal documents share vocabulary, citation formats and
formulaic phrasing, so 800 tokens of *any* legal text may prime
prediction better than 400 tokens restricted to one fragment. Under this
reading, what standard pretraining practice treats as contamination is,
in this domain and at this scale, mild positive transfer.

Note also that the first-64-positions line stays negative throughout
(−0.012 at step 3250). Early positions never stop benefiting from
masking; only later positions flip.

**Unresolved: the `acts` anomaly.** 88.8% of acts windows contain no
separator, so masking should be close to a no-op there, yet it shows
+0.011 with a visibly rising curve. Either the 11% multi-document windows
carry the whole effect disproportionately, or something else is going on.
Recorded as unexplained rather than glossed over.

**Status.** Arm B is at step 3250 of 5700 (GPU surrendered to another
user). The gap has been stable at +0.022 since step 2000, but the run
should be completed for the final number and — given the earlier finding
that validation loss and BERTScore measure different things — for a
generation-quality comparison, which may not follow the loss comparison.

## 2026-09-27 — architecture 2: prefix-LM (masking verified)

**Why this architecture.** Five of the seven phases (2, 3, 4, 5, 7) are
classification or retrieval, where bidirectional context is strictly more
informative than causal. A plain encoder-only MLM would provide that but
cannot generate, so it could not be compared on Phase 1's task at all.
Prefix-LM gives bidirectional encoding over the prompt AND left-to-right
generation over the continuation, so it is comparable on Phase 1 and
useful in Phases 2-7.

**Attention pattern.** keep = key_is_prefix | (causal & both_in_suffix).
Prefix rows attend to the whole prefix and nothing of the suffix; suffix
rows attend to the whole prefix plus earlier suffix positions.

### Design decisions (choices, not defaults — recorded for the viva)

**1. Split distribution: uniform over [0.7T, 0.9T].**
The Phase-1 test supplies a 512-word prompt (~660 tokens) and asks for a
128-word continuation (~166 tokens) — a ratio of ~0.8. Sampling splits
around that ratio means the training regime matches the evaluation
regime. The spread (rather than a fixed 0.8) stops the model from only
ever seeing one prompt length.

Alternatives considered and rejected: uniform over [0, T) produces
degenerate splits (near 0 it is just a causal LM; near T there is almost
no loss signal); a fixed T/2 maximises supervision but drifts furthest
from the test ratio.

**2. Supervised positions: [s-1, T-2] inclusive — the last prefix token
IS included.**
Position t predicts token t+1, so index s-1 is the position that predicts
the first suffix token. That prediction is made with bidirectional context
over the entire prefix — which is exactly the situation at inference,
where the whole prompt is visible when the first continuation token is
generated. Including it makes training match generation. Excluding it
(so that every supervised prediction is uniformly causal) was the
alternative.

**3. Matching tokens processed, not supervised positions.**
At a 0.8 split only ~20% of positions carry loss, against 100% for the
causal decoder: **~5x less supervision per token processed.** Matching
supervised positions instead would require ~5x the tokens, which the
compute budget does not allow. So the arms are matched on tokens, and the
architecture comparison is partly a supervision comparison. This confound
is real and is reported rather than hidden.

### Verification (src/prefix_lm.py, src/test_prefix_lm.py)

- Mask matches a hand-computed 8x8 matrix at split=4: 4x4 all-ones prefix
  block, suffix rows seeing the full prefix plus a lower triangle.
- Degenerate splits behave: split=0 reduces exactly to a causal mask,
  split=T is fully bidirectional (and yields 0 supervised positions,
  handled explicitly rather than dividing by zero).
- Three isolation tests, passing in both fp32/CPU and bf16/CUDA, on both
  the manual and SDPA paths:
  - perturbing a late prefix token DOES move an early prefix output
    (bidirectionality works)
  - perturbing a late suffix token does NOT move an earlier suffix output
    (causality preserved in the suffix) — exactly 0.00e+00
  - perturbing a suffix token does NOT move a prefix output
    (no backward leak into the prefix) — exactly 0.00e+00
- Target alignment proven, not assumed: supervised positions at split=4,
  T=8 are [3,4,5,6]; loss is bit-identical after perturbing unsupervised
  logits (4.191917 -> 4.191917) and changes when a supervised logit is
  perturbed (-> 3.719481). An earlier version of this code had an
  off-by-one here (masking rows >= s instead of >= s-1), which this test
  would have caught.
- Split sampler: 10k draws at T=1024 give min 0.699 / mean 0.799 /
  max 0.899.

**Implementation note.** SelfAttention raises if given no mask when
use_internal_causal_mask=False, rather than silently attending to future
positions. That failure mode would have been invisible in training.

**Relationship to architecture 1.** Architectures 1 and 2 share an
implementation and differ in attention pattern and training objective.
They are substantially different designs in the sense the brief requires
(different information flow, different loss), but the shared code is what
makes the comparison controlled.

**Next:** PrefixGPT wrapper — parameter count must match the decoder
exactly (110,316,288), initial loss over suffix positions should be
ln(32000) ≈ 10.373, then the overfit test (expected to need more steps
than the decoder, given ~5x less supervision per step — a first datapoint
on the supervision cost).

## 2026-09-27 — architecture 2 verified, and a throughput finding

**PrefixGPT checks passed:**
- Parameter count **110,316,288 — identical to the decoder**, transformer-only
  84,953,856 in both. The comparison is controlled by construction:
  _init_weights, configure_optimizers and num_params are imported from GPT
  rather than reimplemented, so initialization and optimization cannot drift
  between arms.
- Initial loss over supervised positions 10.3095 vs ln(32000) = 10.3735.
- Overfit test: 10.27 -> 0.0000, reaching 0.029 by step 100 (the decoder
  needed ~150). Counterintuitive given ~5x less supervision per step;
  plausibly because only ~20% of positions must be memorised and each
  supervised prediction has bidirectional context. **An observation, not a
  conclusion** — memorising 84 tokens is not the same task as learning from
  1.5B, and the supervision deficit should bite differently at scale.

**Instrumentation verified.** At SPLIT_EVAL=0.8, T=1024: 819 NaN positions,
supervised region 818–1022 inclusive = **205 of 1024 positions = 20.0%**,
exactly as designed. nanmean over the saved array reproduces the logged val
loss exactly (8.5931). The 5x supervision deficit is now a measured number
rather than an estimate.

**Reporting choice for prefix-LM val loss.** Reported over *supervised
positions only*, since that is what the objective optimises. This is NOT
directly comparable to the decoder's all-positions loss. Cross-architecture
comparison should use generation metrics (BERTScore on the local eval set),
not raw val loss.

### Throughput: the cost is the 4D mask, not the masking policy

| configuration             | tok/s  | MFU   | vs decoder |
|---------------------------|--------|-------|------------|
| decoder, is_causal=True   | 64,000 | 28.2% | —          |
| decoder + document mask   | 51,600 | 22.8% | −19.4%     |
| prefix-LM                 | 50,700 | 22.4% | −20.8%     |

Document masking and prefix-LM cost almost exactly the same. Both build a
per-sequence (B,1,T,T) mask, which disables SDPA's FlashAttention fast path;
neither penalty is specific to what the mask *means*. **The ~21% is the price
of any per-sequence 4D mask under SDPA**, and it would presumably be
recoverable with FlexAttention's block-sparse kernels (untested).

Measured at attention-only granularity the document mask cost 40.5%, versus
19.4% end-to-end — attention is roughly half a training step's compute at
this scale.

## 2026-09-28 — ablation complete: arm A vs arm B, final

Both arms finished 5700 steps / 1.494B tokens, seed 1899, identical in
every respect except the attention mask.

| | arm A (no mask) | arm B (doc mask) | Δ |
|---|---|---|---|
| **val loss** | **1.72031** | **1.74219** | **+0.0219** |

The gap was +0.0220 at step 3000 and +0.0219 at step 5700 — flat across
2700 steps. At 4x the measured seed variance (0.005) this is a real
effect, not noise. **Document masking is worse on this corpus.**

Per-source, final:

| source   | arm A  | arm B  | Δ       | first 64 | last 64 | slope   | density |
|----------|--------|--------|---------|----------|---------|---------|---------|
| tribunal | 1.5046 | 1.5512 | **+0.0466** | −0.0164 | +0.0953 | +0.1282 | 1.49 |
| acts     | 2.1081 | 2.1103 | +0.0022 | +0.0026  | +0.0064 | +0.0030 | 1.12 |
| high     | 1.7939 | 1.7946 | +0.0008 | −0.0064  | −0.0003 | +0.0055 | 1.99 |
| supreme  | 2.4496 | 2.4492 | −0.0004 | +0.0028  | +0.0009 | −0.0061 | 1.22 |

**The entire aggregate effect is one source.** Weighting tribunal's +0.0466
by its 46.1% mixture share gives 0.0215 — essentially the whole +0.0219.
The other three sources contribute nothing measurable. This is not "masking
hurts"; it is "masking hurts tribunal, and tribunal dominates the mixture."

Both pre-registered predictions refuted, and the pre-registered confound
ruled out:
- **P1** (benefit tracks boundary density): effect ranking is
  tribunal >> acts > high ≈ supreme; density ranking is
  high > tribunal > supreme > acts. No correspondence.
- **P2** (benefit grows with position): the *penalty* grows with position
  instead (slope +0.061 overall, +0.128 tribunal).
- **Early-document confound**: predicted masking would hurt most at low
  position-within-document. The first-64 delta is *negative*
  (−0.010 overall, −0.016 tribunal) — masking helps exactly where the
  confound said it should hurt.

The `acts` anomaly noted at step 3250 (+0.011, unexplained given 88.8%
single-document windows) resolved with training to +0.0022.

## 2026-09-28 — WHY: adjacent documents are not independent

Ran an ordering check on the raw JSON (src/check_ordering.py): are
consecutive documents more similar than random pairs? First 2000 documents
per source, in file order — the order the memmaps preserved.

| source   | density | % adjacent pairs > random p99 | penalty |
|----------|---------|-------------------------------|---------|
| tribunal | 1.49    | **11.3%**                     | **+0.0466** |
| acts     | 1.12    | 5.5%                          | +0.0022 |
| supreme  | 1.22    | 2.1%                          | −0.0004 |
| high     | **1.99**| 1.4%                          | +0.0008 |

The qualitative sample is far more convincing than the Jaccard ratios.
Most-similar adjacent tribunal pairs:
[1676] Purvanchal Vidyut Vitran Nigam Ltd. vs M/S. Jalan Concast Ltd. Rp 2016-07-11
[1677] Purvanchal Vidyut Vitran Nigam Ltd. vs M/S. Jalan Concast Ltd. Rp 2016-07-11
jaccard 1.000
[15] Chhattisgarh SEB vs Raghuvir Ferro Alloys Ltd. 2006-11-28
[16] Chhattisgarh SEB vs Raghuvir Ferro Alloys Ltd. And 2006-11-28
jaccard 0.996

Same parties, same date, jaccard ~1.0 — these are the same matter filed
twice (main appeal and review petition, or near-duplicate records), not
merely "related" documents. Roughly **1 in 9 adjacent tribunal pairs** is
this similar.

Contrast: supreme's most-similar adjacent pair reaches only jaccard 0.276
and shares nothing but a date; high's top pairs are both "Allahabad High
Court 2017" with no shared parties.

### The explanation, and it requires both factors

Document masking is justified by the assumption that packed documents are
independent, so cross-document attention is noise. **In this corpus that
assumption is an empirical property, not a given, and it fails for
tribunal.** When a block packs two consecutive tribunal documents there is
a substantial chance the second is a near-duplicate of the first, so
cross-document attention is reading the *same case* — enormously
predictive. Masking destroys precisely that.

Neither factor alone explains the data:
- **acts** has comparable relatedness (5.5%, and consecutive university
  Acts at jaccard 0.98) but 88.8% single-document windows — there is
  rarely a boundary to mask, so relatedness cannot matter.
- **high** has the highest density (1.99) but the lowest relatedness
  (1.4%) — boundaries everywhere, nothing worth attending across.
- **supreme** has neither.

**penalty ≈ boundary density × neighbour relatedness.** Tribunal is the
only source with meaningful amounts of both.

### What this actually is

Partly a **near-duplicate problem wearing an architecture costume.** The
MD5 dedup in build_memmaps removed 123,603 *exact* duplicates; these
near-duplicates survived it. A corpus with proper near-duplicate filtering
(MinHash/LSH, or simply deduplicating on party names + date) might show no
tribunal penalty at all.

That is a **testable prediction** and the clearest "what we would do next
with more compute": re-run the ablation on a near-duplicate-filtered
corpus. If the tribunal penalty disappears, the finding is confirmed —
document masking is fine, and what it appeared to cost was really the
value of leaked duplicates.

**Reportable claim.** Intra-document attention masking is standard practice
justified by an independence assumption. This corpus shows that assumption
can fail in a way that is invisible unless measured, and when it fails the
technique removes signal rather than noise. The effect is confined exactly
to the source where the assumption breaks. Whether this is a reason to
avoid masking, or a reason to deduplicate harder, is the open question —
and the second reading is more likely correct.

## 2026-09-28 — architecture 2 trained

arch2_prefix: 5700 steps, 1.494B tokens, seed 1899, val 1.78326 over
supervised positions at split 0.8.

**Not comparable to the decoder's 1.72031** — different objective,
different measurement (20% of positions, fixed split). Cross-architecture
comparison must use generation metrics on the local eval set, not raw val
loss.

One within-architecture observation: **tribunal reaches 1.486 under
prefix-LM**, lower than either decoder arm (1.505 / 1.551), while every
other source is worse. Bidirectional prefix context helps tribunal
specifically — the same source that loses most from document masking.
Consistent with tribunal containing near-duplicate material that benefits
from any mechanism giving broader context.

## 2026-09-28 — generation quality REVERSES the ablation (PRELIMINARY)

All three checkpoints scored on the same frozen 200-example eval set,
same decoding config (T=0.8, top_p=0.95, rep_penalty=1.1, EOS suppressed,
seed 0).

| model            | val loss | BERTScore F1 | ROUGE-L | BLEU-4 | distinct-4gram | % short |
|------------------|----------|--------------|---------|--------|----------------|---------|
| arm A (no mask)  | **1.7203** | 0.8187     | 0.1995  | 0.0491 | **0.7503**     | 5.0%    |
| arm B (doc mask) | 1.7422   | **0.8547**   | **0.2924** | **0.1448** | 0.9767  | 1.0%    |
| arch2 prefix-LM  | (n/c)    | 0.8407       | 0.2492  | 0.0945 | 0.9279         | 4.0%    |

**Document masking loses on validation loss by +0.022 and wins on
BERTScore by +0.036.** BLEU-4 nearly triples. The two metrics do not merely
diverge here — they point in opposite directions, and both gaps are far
outside noise (seed variance on val loss is 0.005; the BERTScore gap is
10x the local-vs-leaderboard offset of 0.0034).

### Likely mechanism: arm A degenerates

distinct-4gram is **0.7503 for arm A** against 0.9767 for arm B. Roughly a
quarter of arm A's 4-grams are repeats — it loops. Arm A also produces
125.8 mean words with 5.0% short, versus 128.0 and 1.0% for arm B.

Both failures are exactly what BERTScore punishes: repetition destroys
precision, short output destroys recall. So arm A's deficit is not subtle
quality, it is degeneracy.

**Hypothesis (not concluded).** Arm A learned to exploit cross-document
context — which, per the ordering analysis, includes near-duplicate
tribunal documents where copying from earlier context is highly rewarded.
At generation time there is no duplicate to copy, so the learned copying
behaviour turns inward and the model repeats itself. Masking prevented that
habit from forming. Consistent with everything measured so far, but
untested.

### Why this matters more than either measurement alone

An earlier entry recorded that val loss and BERTScore measure different
things (per-source perplexity varied 2.6x while per-source BERTScore was
flat within 0.3%). This is the stronger version: they **actively
contradict**. The model with worse perplexity generates substantially
better text.

Practical consequence: **selecting a checkpoint or architecture by
validation loss would have chosen the worse generator here.** That is a
concrete answer to the question the brief poses about the relationship
between validation loss/perplexity and generation quality.

### CAVEATS — this is preliminary and must be confirmed

1. **One decoding config, tuned for nothing.** T=0.8/top_p=0.95/rp=1.1 were
   the first values tried, weeks ago. Arm A's named failure is repetition,
   and rep_penalty is the parameter that directly counteracts it. **If arm A
   at rp=1.4 recovers to ~0.85, the finding becomes "arm A needs different
   decoding", not "masking produces better generators."** Different claim,
   still interesting, but it must be tested before anything is concluded.
2. **One generation seed.** Sampling is stochastic; n=1. Re-scoring at a
   second seed is required before the ordering can be trusted.
3. **`high` swings 9 points** (0.8108 -> 0.9039) between arms — the largest
   single effect in the table, and NOT the source where masking affected
   loss (that was tribunal). Unexplained.

### Also: prefix-LM is strong

0.8407 against the causal decoder's 0.8187, **despite ~5x less supervision
per token processed** at matched tokens. Bidirectional encoding of the
prompt appears to help this task materially — which is plausible, since the
model reads all ~660 prompt tokens in both directions before generating.

Its per-source spread is much wider than the decoder's (0.8269–0.8716, a
5.4% range, vs the decoder's 0.3%), with `high` again the outlier on the
high side.

The ratio-mismatch concern was unfounded: prefix-LM trained on splits of
0.7–0.9 but generates at ratios traversing 1.0 -> 0.75, and the sample
output is fluent, well-formed statutory prose with correct enumeration and
cross-referencing. Extrapolation above the training range is fine.

### Next

- Submit arm B (best measured model; 0.8547 local vs the 0.8260 that
  submission 1 scored). Also re-validates the local evaluator at a
  different score level.
- Re-score arm A and arm B at a second generation seed.
- **Decoding sweep on arm A specifically** — this is the check that could
  overturn the interpretation above.

## 2026-09-29 — CONTAMINATION FOUND, and the headline result is revised down

### Detection

The local evaluator over-predicted arm B by 0.0208 against 0.0034 for the
decoder, which suggested the eval set favoured one model. Hypothesis: eval
documents have near-duplicates in training that exact-match MD5 dedup missed.

Method (src/check_contamination.py): neighbour search rather than all-pairs.
The ordering analysis established that near-duplicates are ADJACENT in the
source files, and build_heldout sampled past a byte offset, so a held-out
document's twin is almost certainly a few positions away in the same file.
Streamed each file with a rolling 21-document buffer; whenever the centre was
a held-out document, compared its 20 neighbours (which ARE in training)
against every eval window from that source, on 5-word shingle Jaccard.

**Limitation, stated:** this cannot find a duplicate far from its twin in file
order. Given the ordering analysis that is where they live, but the check is
not exhaustive. (MinHash/LSH was the intended method; `datasketch` could not
be installed — the campus network intercepts HTTPS to pypi.org with an IITK
certificate.)

### Result: the contamination is confined to one source

| source   | median best-match | >= 0.5   | max   |
|----------|-------------------|----------|-------|
| **high** | **0.413**         | **21/50**| 0.867 |
| acts     | 0.024             | 0/50     | 0.178 |
| tribunal | 0.006             | 0/50     | 0.318 |
| supreme  | 0.005             | 0/50     | 0.066 |

**42% of High Courts eval windows have a near-duplicate in training**; the
other three sources are essentially clean (medians < 0.03). 179 of 200
examples survive a threshold of best-match Jaccard < 0.5.

This aligns exactly with where the arm A/arm B gap was largest: `high` showed
+0.093 on the full set against +0.014–0.020 for every other source. Precisely
the pattern contamination would produce if arm B is the better copier.

### Re-scored on the clean subset — the effect is real but ~60% smaller

| | full set (200) | clean set (179) |
|---|---|---|
| arm A | 0.8187 | 0.8255 |
| arm B | 0.8547 | 0.8408 |
| **gap** | **+0.0360** | **+0.0153** |

Per-source gaps:

| source   | full set | clean set |
|----------|----------|-----------|
| high     | +0.0931  | **+0.0422** |
| acts     | +0.0196  | +0.0195   |
| supreme  | +0.0173  | +0.0012   |
| tribunal | +0.0141  | +0.0098   |

`high` halves, exactly as predicted. `acts` is unchanged (it was clean).
`supreme` collapses to nothing.

The inflation of 0.021 is almost exactly the 0.0208 by which the local score
over-predicted Codabench on arm B — an independent corroboration that this is
the right explanation for that discrepancy.

### What survives, and what is withdrawn

**Withdrawn:** the claim that document masking improves BERTScore by 0.036.
Roughly 60% of that was memorised near-duplicates. Earlier entries stating
0.036 as the headline figure are superseded by this one.

**Survives:** document masking improves generation quality by **+0.0153
BERTScore** on uncontaminated examples. Supporting evidence:
- The repetition diagnostics are **untouched** by the filtering — arm A
  0.7825 distinct-4gram, arm B 0.9797 on the clean set, essentially identical
  to the full-set values (0.7503 / 0.9767). Arm A's degeneracy is real and has
  nothing to do with contamination.
- The direction matches the leaderboard (0.8260 -> 0.8339).
- The decoding sweep showed arm A cannot be tuned up to arm B.

**Caveat on magnitude:** +0.0153 is only ~4x the generation-seed variance of
0.0035, against ~10x for the original figure. "Real but modest" rather than
"large". The clean-subset comparison should be repeated at a second seed
before the number is quoted as settled.

**Caveat on the clean subset:** it is unbalanced (29 high / 50 / 50 / 50),
since only `high` lost examples. Per-source figures remain valid; the
aggregate is weighted slightly differently from the full set. Adding
replacement `high` examples was considered and rejected — it would invalidate
every measurement recorded since 26 September, all of which used the frozen
200-example set.

### Methodological note for the report

Exact-match deduplication is insufficient on this corpus. MD5 removed 123,603
duplicates during memmap construction, yet 42% of one source's held-out
windows still had near-twins in training. **A held-out set built by excluding
exact hashes is not a held-out set.** This is the single most consequential
methodological finding of the project so far, and it was only caught because
the local/leaderboard discrepancy was large enough on one model to demand an
explanation.

Implication for the ablation's own interpretation: the document-masking
penalty on validation loss was attributed to near-duplicate tribunal documents
providing genuine cross-document signal. That explanation is unaffected — it
concerns training-set structure, not eval contamination — but both findings
now point at the same root cause, which is that this corpus contains
substantial near-duplicate material that standard dedup does not remove.

## 2026-10-01 — architecture 4: encoder-decoder with span corruption (design)

### Why an encoder-decoder for the fourth architecture

Architectures 1, 2 and 3 all share one transformer stack and differ only in
what attends to what (full causal / bidirectional prefix / windowed). A
separate encoder and decoder joined by cross-attention is the only remaining
family with genuinely different information flow, which is what the brief
means by "substantially different". Phase 6 (abstractive summarization) is
also the natural home for it, though that is seven weeks away and the weaker
reason.

### Layer split: 4 encoder + 6 decoder

Parameter arithmetic at d=768:
- embeddings + 2 final norms = 25,363,968
- encoder block (self-attn 4d² + FFN 8d² + 2 norms) = 7,079,424
- decoder block (+ cross-attn Q 589,824, KV 1,179,648, proj 589,824, + 1 norm)
  = 9,439,488

| split | total params | vs baseline 110,316,288 |
|-------|--------------|-------------------------|
| 4 enc + 6 dec | 110,318,592 | +2,304 (0.002%) |
| 8 enc + 3 dec | 110,317,824 | +1,536 (0.001%) |

**Chose 4+6, not the marginally closer 8+3.** The 768-parameter difference
between the options is not a real criterion; the choice should be
architectural.

Reasoning: Phase 1 scores *generation*, and the decoder is what generates.
The other three architectures each have 12 layers of generative depth; a
3-layer decoder would compete with a quarter of that while producing 128
consecutive words. 4+6 is also closer to standard practice — T5 uses equal
stacks.

**The rejected alternative, recorded because it is defensible.** 8+3 would
suit Phases 2–7, where classification heads attach to the encoder and
generation is irrelevant. Two reasons it lost: Phase 1 is the measurement
actually in hand, and prefix-LM already provides bidirectional encoding for
downstream phases without sacrificing generative depth. The encoder-decoder
does not have to be the encoder specialist.

### Span corruption: 15% corruption, mean span length 25

**This is a deliberate deviation from T5, which uses mean span 3.**

The justification is task match, not convenience: the Phase 1 objective is
generating 128 *consecutive* words. A model trained to fill single-token gaps
is optimised for a different thing than one trained to reconstruct long
contiguous clauses from surrounding context. Span 25 trains the behaviour the
evaluation asks for.

It also happens to fit the sentinel budget — 15% of 1024 tokens is ~154
masked tokens, which at span 25 is ~6 spans, within the 10 reserved
`<|unused0-9|>` tokens. But that is a constraint satisfied, not the reason.

**Cost, stated:** at 15% / span 25 the decoder target is only ~160 tokens, so
the decoder trains on far shorter sequences than the 1024 the other three
architectures see every step. Whether that handicaps generation is an open
question the comparison will answer.

**Sentinels and EOS kept separate.** `<|endoftext|>` (token id 0) serves as
decoder start and EOS; `<|unused0-9|>` are sentinels only. An earlier draft
reused the last sentinel as EOS, conflating two roles.

### Implementation notes (issues found in the first draft)

- Non-overlap in span selection was checked by distance between *start
  points*, which does not prevent overlap when spans are longer than the
  gap. Construct spans left-to-right by alternating span and gap lengths —
  non-overlap holds by construction, as in T5.
- The encoder must use `SelfAttention` from prefix_lm.py, not
  `CausalSelfAttention` — the latter holds an internal causal mask buffer
  that always applies, so setting a config flag it does not read would leave
  the "encoder" causal and the bug would be invisible.
- Encoder inputs and decoder targets have *variable length* (depending on how
  many spans were sampled), so batching needs padding, and cross-attention
  must mask padded encoder positions. This is the main structural difference
  from every data path built so far.
- Residual rescaling by 1/sqrt(2*n_layer) must be applied as in the other
  architectures, or initialization differs and the comparison is not matched.

### Open question: Phase 1 generation

A model trained to fill sentinels does not naturally continue text. Planned
approach: encoder receives `prompt + <X>`, decoder generates the fill for
`<X>`. That matches the training format and is what T5 does for zero-shot
continuation. **If this does not produce fluent output the architecture
cannot be scored on Phase 1 and the four-way comparison loses a leg** — so it
is worth testing early, before committing 8 hours of training.

## 2026-10-01 — architecture 4 built and verified

### Generation feasibility was tested BEFORE training

A span-corruption model does not naturally continue text, so the Phase 1
format (encoder gets `prompt + <X>`, decoder produces the fill) was tested on
a 2-layer overfit model first, to avoid discovering an unusable architecture
after 8 GPU hours.

**Mechanics pass**: the decoder reproduces a memorised fill exactly, starts
with the correct sentinel, and emits EOS.

**The length question could not be answered this way.** A model overfit on one
example reproduces that one example regardless of input, so its 8-token output
says nothing about span-length behaviour. The risk was assessed by reasoning
instead, and mitigated pre-emptively.

### Corruption scheme: three deliberate deviations from T5

**1. Mean span 25, not 3.** Phase 1 asks for 128 *consecutive* words.
Training the model to reconstruct long contiguous clauses matches the
evaluation; filling single-token gaps does not.

**2. Terminal spans (p=0.25).** The final span sometimes runs to the end of
the sequence, so the model sees sentinels with NO RIGHT CONTEXT during
training — exactly the inference condition. Without this, the Phase 1 format
would never appear in training.

**3. Long-terminal mode (p=0.15): one span of 120–200 tokens at the end.**

Mode 3 was added after measurement showed modes 1–2 could not reach the
required length. Under standard span corruption the entire mask budget at
p=0.15 on 1024 tokens is 150 tokens shared across ~6 spans, so **no single
span can approach the ~166 tokens Phase 1 needs**. Measured: p99 span length
78, max 117 over 2000 sequences. Raising the length variance alone cannot fix
this — the budget is the binding constraint.

Resulting mix: ~15% long-terminal, ~24% short-terminal, ~61% standard.
Span-length distribution after the fix: p10 8, median 22, p90 52, p99 170,
max 200. 38% of long-terminal spans reach 166 tokens.

**None of this is retrofittable after training**, which is why it was measured
and fixed first.

### Architecture: 4 encoder + 6 decoder = 110,318,592 params

+2,304 over the baseline decoder (0.0021%). The marginally closer 8+3 split
(+1,536) was rejected — 768 parameters is not a criterion. Phase 1 scores
generation and the decoder generates; a 3-layer decoder would compete against
12 layers of generative depth in the other three architectures. 4+6 is also
closer to T5's equal stacks. 8+3 would have suited Phases 2–7, but prefix-LM
already supplies bidirectional encoding for those without sacrificing
generative depth.

Verified: cross-attention handles T_enc != T_dec; padded encoder positions do
not reach the decoder (exactly 0.00e+00 under perturbation); decoder
self-attention is causal (0.00e+00 before the perturbed position); overfit
test converges to 0.0000.

Initial loss 10.6067 vs ln(32000)=10.3735 — same pattern as the efficient
architecture (10.513 mean over 5 seeds, spread 0.102). Both assemble modules
in a different order from GPT.__init__, so the same seed yields a different
draw sequence. **The arms are matched in architecture, data and budget, not in
initialization.**

### The supervision deficit is the worst of the four

Measured on a real batch: **1,227 supervised decoder positions for 8,192
tokens read — 6.7x less supervision than a causal decoder** (prefix-LM was
~5x). Padding waste is small: encoder 98.8% real tokens, decoder 97.7%.

Matching is on **tokens read by the encoder** (1024/sequence, same as every
other architecture), not supervised positions. Matching supervision would need
~6.7x the tokens, which the budget does not allow. Reported as a confound.

Consequence: `tokens_seen` and MFU for this architecture count encoder tokens,
so **its MFU figure is not comparable** to the other three.

### Data pipeline

Span corruption happens on the fly per batch item — pre-corrupting would make
every pass see identical corruptions, defeating the objective. Batches pad to
the batch maximum rather than sorting by length, since sorting would correlate
batch composition with sequence length and interact badly with the
source-mixture sampler.

**Eval reproducibility required resetting two RNGs, not one.** The corruption
RNG and the window-sampling RNG in MemmapDataLoader are independent; seeding
only the first still produced different val batches. Both are reset once at
the top of the eval loop (not inside get_batch, which would make all 20 val
batches identical rather than making the same 20 reproducible).

Per-position loss curves are not saved for this architecture — decoder targets
are variable-length, so the position axis means something different from the
other three.

### Data-quality note

A randomly sampled training window decoded to mojibake — binary or
mis-encoded content that survived PDF extraction into the memmaps. Byte-level
BPE encodes anything, so such documents pass through silently. Prevalence
unmeasured. It affects all four architectures equally so the comparison is
unthreatened, but it is a corpus-quality observation worth reporting.

## 2026-10-01 — all four architectures trained and compared

Matched: 5,700 steps / 1.494B tokens read, seed 1899, same tokenizer, same
mixture, same LR schedule. Scored on the contamination-filtered 179-example
eval set, T=0.8 / p=0.95 / rp=1.1 / EOS suppressed / seed 0.

| architecture     | params      | val loss | BERTScore | ROUGE-L | BLEU-4 | distinct-4gram | % short |
|------------------|-------------|----------|-----------|---------|--------|----------------|---------|
| decoder          | 110,316,288 | 1.7203   | 0.8255    | 0.1995  | 0.0491 | 0.7825         | 5.6%    |
| encoder-decoder  | 110,318,592 | 2.4309*  | 0.8311    | 0.1946  | 0.0320 | **0.9906**     | **0.0%**|
| efficient        | 110,316,288 | 1.7200   | 0.8328    | 0.2246  | 0.0629 | 0.8270         | 3.9%    |
| **prefix-LM**    | 110,316,288 | 1.7833*  | **0.8358**| 0.2198  | 0.0581 | 0.9283         | 2.8%    |

\* not comparable across architectures: prefix-LM measures ~20% of positions
at a fixed 0.8 split; encoder-decoder measures span reconstruction, a harder
task than next-token prediction.

**Prefix-LM wins the matched comparison**, +0.0103 over the plain decoder.
Bidirectional encoding of the prompt is worth more than any other
architectural change tried.

For reference, outside the matched set:
- arm B (decoder + document masking, 1.5B): **0.8408** — still the best model
  overall, and it is an ablation on architecture 1, not a fifth architecture
- prefix-LM at 3.0B: 0.8391

### Supervision differs enormously and is not controlled

| architecture | supervised positions per 1024 tokens read |
|---|---|
| decoder / efficient | 1024 |
| prefix-LM | ~205 (5x less) |
| encoder-decoder | ~153 (6.7x less) |

Matching is on **tokens read**, since that is what the compute budget buys.
Prefix-LM therefore wins *despite* receiving a fifth of the decoder's
supervision, which strengthens rather than weakens the result. Matching on
supervised positions instead would require 5–7x the tokens for those two
architectures, which the budget does not allow.

### The encoder-decoder result is informative in a specific way

It ranks **third on BERTScore but LAST on ROUGE-L and BLEU-4** (0.1946 /
0.0320, below even the plain decoder), while having by far the cleanest
generation statistics: 0.9906 distinct-4grams, 0% short, exactly 128 words
every time.

That combination means it produces fluent, non-repetitive, correctly-sized
legal prose that does not match the reference's actual wording. Which is what
a span-reconstruction objective would be expected to give: it learned to
generate plausible text for a gap, not to continue a specific passage.

**This is the risk flagged at design time materialising.** The long-terminal
corruption mode (15% of examples, 120–200 token spans at the sequence end) was
added specifically to rehearse the Phase 1 format, and it was enough to make
generation *work* — fluent output, correct length, no degeneracy — but not
enough to make it *match*. The objective mismatch shows up in exactly the
metrics that reward lexical overlap.

Worth noting it is also the fastest to train by a wide margin: **122,713 tok/s
at MFU 53.7%**, roughly double the decoder's 63.6k, because the decoder stack
processes ~157 positions instead of 1024. (That MFU counts encoder tokens and
is not comparable to the others.)

### CORRECTION: the repetition pattern is not universal

An earlier entry noted a monotonic relationship — less repetition, better
BERTScore — across decoder (0.783 / 0.8255), efficient (0.827 / 0.8328),
prefix-LM (0.928 / 0.8358) and arm B (0.980 / 0.8408).

**The encoder-decoder breaks it.** It has the least repetition of any model
(0.9906) and ranks third. So the relationship holds *within the decoder
family*, where repetition was the dominant failure mode, but it is not a
general law. Low repetition is necessary for a good score here, not
sufficient.

Recording the correction because the three-point version was the more
appealing claim and would have been wrong.

### Per-source

| architecture | acts | high | supreme | tribunal |
|---|---|---|---|---|
| decoder | 0.8206 | 0.8188 | 0.8307 | 0.8290 |
| encoder-decoder | 0.8294 | 0.8374 | 0.8296 | 0.8306 |
| efficient | 0.8290 | 0.8387 | 0.8317 | 0.8341 |
| prefix-LM | 0.8269 | 0.8526 | 0.8331 | 0.8377 |

`high` has the widest spread across architectures (0.8188–0.8526) and is where
prefix-LM gains most. It is also the source with the shortest documents and
the one that was contaminated in the unfiltered eval set — so its behaviour
remains the least well understood of the four.

### Outstanding

- 4th Codabench submission (have 3, need 4 by Oct 6)
- clean-subset arm A vs arm B at generation seed 1, to confirm the +0.0153
  ablation gap (only ~4x seed variance, so one seed is thin)
- prefix-LM + document masking at 3B — the configuration the evidence points
  at, combining the winning architecture with the winning ablation

## 2026-10-02 — prefix-LM + document masking: the interventions do NOT compose

prefix_docmask_3B, 11,400 steps / 2.99B tokens, seed 1899.

| model | val loss | BERTScore (clean) | distinct-4gram |
|---|---|---|---|
| prefix-LM 3B | 1.6560 | 0.8391 | 0.9539 |
| prefix-LM 3B + doc mask | 1.7125 | **0.8392** | 0.9731 |

**+0.0001 BERTScore for +0.0565 val loss.** A null result: the difference is
a thirtieth of the measured generation-seed variance (0.0035).

Document masking gave +0.0153 BERTScore on the plain decoder but nothing here.
Hypothesis: the decoder's gain came from suppressing degenerate copying
(distinct-4gram 0.78 -> 0.98). Prefix-LM at 3B had already largely avoided
that habit (0.9539), so masking pushed repetition lower still (0.9731) but
repetition was no longer the binding constraint. Consistent with the earlier
correction that low repetition is necessary, not sufficient.

**The val-loss penalty replicated and strengthened.** +0.0565 here vs +0.0219
on the decoder, and again concentrated almost entirely in tribunal (+0.110 vs
+0.047), the source with near-duplicate structure. Independent confirmation of
the mechanism on a second architecture.

**Inference-only bug found by the assert.** The combined mask produced
fully-masked query rows during generation but never in training: left-padded
prompts mean a padded query's causal row covers only other padded columns.
Fixed by OR-ing the identity so every query attends to at least itself. Would
have produced NaN, not an error.

# PHASE 2 — Legal Named Entity Recognition

## 2026-10-08 — data profile

70 training documents (fold1 + fold2, 35 each), 35 test documents (fold3),
8,393 gold entities, 12 labels. **This is three orders of magnitude smaller
than Phase 1's 3.5B tokens.**

Two consequences that shape everything else:

**1. Full fine-tuning of a 110M model on 70 documents will overfit almost
immediately.** The brief requires five different adaptation techniques plus
LORMA; parameter-efficient methods are therefore not merely a requirement but
plausibly the better choice on their merits. Worth stating as a prediction
now, before the experiments, so the result can confirm or refute it.

**2. Every document overflows the context window.** 100% of training documents
exceed 1024 tokens (median 5,542, p90 9,961, max 23,781); 94.3% of test
documents do. Windowing is mandatory, not an optimisation. The brief lists
"document and sentence context windows" as something to investigate, so this
is intended.

### Label distribution — macro-F1 makes the rare labels decisive

| label | count | share |
|---|---|---|
| DATE | 1,575 | 18.8% |
| STAT | 1,260 | 15.0% |
| PREC | 1,183 | 14.1% |
| COURT | 1,013 | 12.1% |
| AUTH | 1,003 | 12.0% |
| CASENO | 760 | 9.1% |
| APP | 407 | 4.9% |
| RESP | 381 | 4.5% |
| JUDGE | 245 | 2.9% |
| A.COUNSEL | 195 | 2.3% |
| WIT | 187 | 2.2% |
| R.COUNSEL | 184 | 2.2% |

An 8.6x range between the most and least frequent. **Under macro-F1,
R.COUNSEL (184 examples) is worth exactly as much as DATE (1,575).** Four
labels have under 250 examples and together carry a third of the metric.

Note also that WIT, APP and RESP are all person names distinguished only by
narrative role — e.g. WIT `'Mange Ram'` vs APP `'Jage Ram'` in the same
document. That discrimination needs context, not surface form.

Entities: median 105 per document, p90 195, max 403. Spans are short — median
13 characters / 4 tokens, p90 33 chars / 10 tokens.

**Annotation sanity: 0 out-of-range spans, 0 overlapping spans, and all 8,393
`entity_text` fields match `text[start:end]` exactly.** So the offsets are
plain Python character indices and the alignment problem is well-posed.

## 2026-10-08 — tokenizer alignment ceiling: 98.66% macro

Strict evaluation means a one-character error scores zero, so the first
question is what fraction of gold spans the Phase-1 tokenizer can represent at
all. Measured by round trip: character offsets -> token indices -> character
offsets.

| convention | exact | % |
|---|---|---|
| strict (span must align to token boundaries) | 307 | 3.66% |
| expand (take every token the span touches) | 307 | 3.66% |
| **trim (expand, then strip whitespace)** | **8,240** | **98.18%** |
| trim_punct (trim, then strip trailing punctuation) | 7,937 | 94.57% |

**`trim` is the convention. Ceiling on macro-F1 is 98.66%.**

The strict/expand collapse to 3.66% is the byte-level BPE leading-space
convention: " the" is one token spanning the space and the word, so almost no
entity start coincides with a token boundary. Trimming whitespace recovers
nearly everything.

### A plausible hypothesis, refuted by the data

Every one of the 12 sampled failures had the shape **"start ok, end late"** —
the final token extending past the gold end, often by trailing punctuation
(`'Ors.'` -> `'Ors.['`, `'SCC 47]'` -> `'SCC 47].'`, `'1063)'` -> `'1063);'`).
Stripping trailing punctuation looked like a free win, particularly for PREC
(the worst label at 93.15%).

**It is not.** 397 gold spans legitimately END in punctuation — 217 of them
PREC, where the citation bracket is part of the entity (`'[AIR 1984 SC 1728]'`).
Stripping breaks all of those, and `trim_punct` scores 3.6 points WORSE
(94.57% vs 98.18%). The count of punctuation-ending gold spans was computed
before running the comparison, and correctly predicted the outcome.

### What the irreducible 1.3% consists of

Two kinds, both at the END of spans (starts are never wrong):

- **Punctuation merged into the final token** — fixable only by a rule that
  breaks 397 other spans.
- **Merged suffixes** — `'Board'` -> `'Boards'`, `'Constitution'` ->
  `'Constitutional'`, `'State Government'` -> `'State Governments'`. One BPE
  token spans both the entity and the extra characters, so no character-level
  rule can split it without knowing the answer.

Per-label ceilings under `trim`: PREC 93.15% (worst — long citations ending in
brackets), STAT 98.10%, RESP 98.16%, COURT 98.22%, APP 98.28%, AUTH 98.60%,
A.COUNSEL 99.49%, CASENO 99.87%, and DATE / WIT / JUDGE / R.COUNSEL at 100%.

**Implication for the report:** a tokenizer trained for language modelling
imposes a measurable ceiling on a downstream span task, and that ceiling is
unevenly distributed across entity types. PREC loses 6.85% before any model
runs. A task-specific tokenizer, or character-level boundary refinement, is
the obvious remedy — worth noting as future work rather than attempting now.

### Next

- BIO tagging and its inverse, round-trip tested against the 98.18% baseline
- windowing strategy for documents that are 5x the context window
- validation split (no dev set is provided; 70 documents total)

## 2026-10-08 — baseline: full fine-tuning of prefix-LM 3B

First L-NER run. arch2_prefix_3B, token-classification head, 10 epochs,
batch 8, lr 3e-5, no class weighting. Cross-validated across the two folds.

| val fold | macro-F1 | best epoch |
|---|---|---|
| fold2 | 0.2665 | 8 |
| fold1 | 0.2541 | 5 |
| **mean** | **0.2603** | spread 0.0123 |

Fold agreement is good (0.012), consistent with check_folds finding the folds
comparable and free of cross-fold near-duplicates.

**Overfitting is severe and fast, as predicted.** Train loss 1.2303 -> 0.0047
over 10 epochs while validation loss bottoms at epoch 3 (0.1441) and climbs
steadily thereafter. 56 training documents against 110M parameters. This is
the registered prediction confirmed and the case for parameter-efficient
methods.

**Bidirectionality verified empirically**: with split_indices = T, perturbing
token 100 changes the representation at token 20 by 1.59e-01. The causal
models give exactly 0.00e+00 on the same test.

## 2026-10-08 — error analysis: the problem is decoding, not finding

Breakdown against fold1 gold (4,120 entities, 5,869 predicted — 1.42x):

| category | n | % | |
|---|---|---|---|
| exact | 1,864 | 45.2% | correct |
| **boundary** | **1,114** | **27.0%** | right label, wrong span |
| label | 71 | 1.7% | right span, wrong label |
| both | 296 | 7.2% | overlapping, both wrong |
| missed | 775 | 18.8% | no overlapping prediction |
| **spurious** | **2,524** | **43.0%** | predicted where nothing is |

**27% of gold entities were found with the correct label and scored zero
because of the span.** That is the largest addressable bucket, and larger than
any plausible architectural gain.

### The boundary errors are truncation of long spans

Direction is overwhelmingly one-sided — PREC shows 236 "end early" against 45
"end late", 124 "start late" against 13 "start early". Worked examples:
gold 'State of Maharashtra v. Balram Bama Patil, (1983) 2 SCC 28'
pred 'State of Maharashtra v. Balram Bama Patil, (1983)'
gold 'Girija Shanker v. State of U.P.(2004) 3 SCC 793'
pred 'Girija Shanker v.'
gold 'R. Prakash v. State of Karnataka (2004) 9 SCC 27'
pred 'R'


**Control: 785 of 815 of these gold spans round-trip exactly through the
tokenizer.** So this is not a decoding bug and not a tokenizer limit — the BIO
chain is breaking mid-entity, and bio_to_spans terminates the span there. A
long citation needs ~15 consecutive correct token decisions and gets ~12.

The spurious predictions corroborate: `'J'` (from 'Md. Jahidul'), `'AN'` (from
'CAN 7461'), `'C'`, `'Vs'` — fragments, mostly orphan I-tags.

## 2026-10-08 — decoding alone: 0.2541 -> 0.3589 (+41% relative)

No retraining. Model output (merged tag sequences) saved once, then decode
variants compared on it.

| | lenient | strict orphan-I |
|---|---|---|
| no bridge | 0.2541 | 0.3076 |
| + bridge 3 | 0.3119 | 0.3525 |
| + bridge 3 + min 3 chars | 0.3240 | **0.3583** |

**The two fixes are complementary, and the first run missed this by not
crossing them.** Strict orphan-I handling removes fragment predictions
(5,869 -> 3,427 predictions, precision 0.223 -> 0.338); bridging repairs
broken chains (recall 0.291 -> 0.332). Together they beat either alone,
because strict removes the noise that bridging would otherwise extend.

Bridging saturates at gap 3 (0.3525 / 0.3523 / 0.3511 for 3 / 4 / 5), so the
chain breaks are short — one or two tokens.

Confidence thresholding is not worth it: +0.0006 at 0.3, and at 0.5 it
destroys WIT entirely (F1 0.0000) while the aggregate still looks respectable
at 0.3564. **A variant can win on macro-F1 while zeroing a label**, which is
why the comparison now flags zeroed labels explicitly.

**Settled decode: strict orphan-I, bridge 3, minimum 3 characters.** Best
single-fold macro-F1 0.3589.

### Implication for the next experiment

Bridging helping monotonically is direct evidence that **a CRF head should
help**, since it enforces valid BIO transitions during inference rather than
patching them afterwards. That is now an evidence-backed experiment rather
than a guess, and it satisfies the brief's "different prediction heads" item.
A span-based head (enumerate and classify candidate spans) would address the
same failure more completely, since the span is predicted atomically and
cannot fragment.

## 2026-10-08 — where the remaining headroom is

Per-label F1 under the best decode:

| label | F1 | | label | F1 |
|---|---|---|---|---|
| DATE | 0.9332 | | JUDGE | 0.2900 |
| COURT | 0.7296 | | AUTH | 0.2404 |
| STAT | 0.6376 | | A.COUNSEL | 0.2282 |
| CASENO | 0.5869 | | R.COUNSEL | 0.1429 |
| PREC | 0.3052 | | APP | 0.1030 |
| | | | RESP | 0.0948 |
| | | | **WIT** | **0.0156** |

**The split is between form-recognisable and role-dependent entities.** Dates,
court names, statutes and case numbers have distinctive surface forms and
score 0.59-0.93. APP, RESP and WIT are all person names; which one a name is
depends on its role in the narrative, not on how it looks. Those three plus
R.COUNSEL average 0.11 and carry a third of macro-F1.

The confusion matrix confirms it: R.COUNSEL -> A.COUNSEL 55 times,
A.COUNSEL -> R.COUNSEL 22, RESP -> APP 19, APP -> RESP 16, RESP -> WIT 17.
The model finds the names and cannot assign the roles.

Bidirectional attention alone does not solve this — prefix-LM at split=T
already sees full right context. The disambiguating signal is structural:
position relative to "Versus", the `(A-1)` / `(PW-6)` markers, proximity to
"appellant"/"respondent" headers. Whether that needs features, a two-stage
approach, or simply more context is the open question.

## 2026-10-08/09 — CRF head: the largest single gain (+0.18 CV)

Evidence-driven rather than guessed. The error analysis showed BIO chains
breaking mid-entity, and post-hoc gap bridging improved macro-F1 monotonically
through gap 3. A CRF enforces valid transitions during inference instead of
patching them afterwards.

| configuration | CV macro-F1 |
|---|---|
| plain head, lenient decode | 0.2603 |
| plain head, tuned decode (strict + bridge 3 + min 3) | 0.3583 (fold1) |
| **CRF head, Viterbi decode** | **0.4438** |

The CRF beats hand-tuned decode patching by +0.085, so it is not merely doing
the same job more cleanly. Precision rises from 0.218 to 0.515 — it mostly
eliminates invalid predictions rather than finding new entities.

Implementation verified before use: the forward algorithm matches brute-force
enumeration to 4.8e-07, Viterbi matches brute-force exactly, padding leaves
the partition function unchanged, and constrained decoding never emits an
invalid transition. **44% of the 625 tag-pair transitions are structurally
impossible and are hard-masked to -1e4 rather than left to be learned** — with
56 training documents there is not enough signal to learn rules known a priori.

Both CV folds peaked at epoch 9 of 10.

## 2026-10-08 — position features: REFUTED

Hypothesis: 44% of APP and 38% of RESP appear in the first 10% of the
document, where the case header sits, but windows are processed independently
so the model cannot tell where it is. Adding normalised document position
(16 sinusoidal features concatenated before the head) should help the
header-clustered labels and not WIT (median position 0.31, narrative context).

**Result: 0.2578 vs 0.2603 baseline — no effect, and the per-label prediction
failed in detail.** APP got WORSE (0.0565 -> 0.0291) despite being the most
header-clustered label; WIT improved slightly despite the prediction that it
would not.

The flaw in the reasoning, visible in hindsight: APP and RESP have nearly
identical position distributions (median 0.15 vs 0.19). **A feature that
assigns both the same value cannot separate them.** What distinguishes them in
the header is ORDER around "Versus", which normalised position does not
capture. Fold spread also doubled (0.0289 vs 0.0123), so the small difference
is not even reliably negative.

## 2026-10-09 — LoRMA+ (compulsory): 93% of full fine-tuning at 0.28% of parameters

LoRMA is multiplicative where LoRA is additive: h = I(BA) W0 x, with additive
inflation I(BA) = (a/r)BA + I_d so the adapter begins as the identity.

**Fused-QKV deviation, resolved faithfully.** The paper adapts the query and
value projections separately, but this codebase uses a single fused
c_attn: Linear(d, 3d). Applying one transform to the fused output would adapt
Q, K and V jointly under a shared (3d x 3d) transform — a different method. So
the output is sliced and separate transforms applied to the Q and V thirds,
leaving K untouched.

Verified: exact identity at initialisation (0.00e+00 difference from the base
layer), K provably untouched, merged weights reproduce the adapted forward
pass to 8.3e-07, gradients reach only the adapters.

| | CV macro-F1 | leaderboard | trainable |
|---|---|---|---|
| full fine-tuning + CRF | 0.4438 | 0.5161 | 110,336,188 |
| **LoRMA+ (r=8, a=16) + CRF** | **0.4127** | **0.4839** | **314,137 (0.284%)** |

**93% of CV performance and 94% of leaderboard performance from 0.28% of the
parameters.** Both estimates agree, which is the answer to the brief's
question about whether parameter-efficient methods can approach full
fine-tuning here.

Both folds again peaked at epoch 9, then degraded and oscillated (fold2 fell
from 0.407 to 0.311 by epoch 25, swinging between 0.23 and 0.37) while train
loss stayed flat — instability from the 5e-4 learning rate rather than
overfitting. Epoch 1 scores exactly 0.0000, confirming the adapter really does
start as a no-op.

## 2026-10-09 — knowledge distillation (compulsory): the efficiency claim holds, the distillation claim does not

Teacher: full fine-tuned CRF prefix-LM. Student: the same architecture
truncated to 6 of 12 layers, initialised from every other teacher layer
(DistilBERT's scheme). Simplification stated: only the EMISSIONS are distilled
(KL on pre-CRF logits); matching CRF marginals would need forward-backward.

**A contaminated first run was caught and discarded.** The teacher had been
trained on all 70 documents including the validation fold, and scored 0.9833
on fold2 — memorisation, not performance. The students inherited those weights
and reported 0.73 / 0.71. All of it unusable. Re-run with the fold-specific
teacher (trained on fold1 only):

| model | params | macro-F1 (fold2) | ms/batch |
|---|---|---|---|
| teacher, 12 layers | 110,336,188 | 0.4520 | 38.1 |
| student, distilled | 67,859,644 | 0.4273 | 19.1 |
| student, hard labels only | 67,859,644 | 0.4237 | 19.1 |

**94.5% of teacher macro-F1 at 62% of parameters and 2.0x faster inference.**
That answers the brief's efficiency question directly.

**But the distillation effect is +0.0036 — nil.** Epoch-to-epoch swings in both
runs are 0.03-0.05, an order of magnitude larger. The honest conclusion is that
**the student retains performance because it was initialised from the teacher's
weights, not because of the KL term.** Layer truncation plus fine-tuning does
the work; soft targets add nothing measurable.

Without the hard-label control run, this would have been reported as
"distillation retains 94.5%" and attributed to the wrong mechanism. The control
is what makes it an experiment.

Plausible reason: with 56 documents and a student starting from the teacher's
own weights, the student's emissions are already close to the teacher's, so KL
has little to correct. A student NOT initialised from the teacher would be the
test — not attempted within the deadline.

## 2026-10-10 — decoder vs prefix-LM: my registered prediction was WRONG

**Registered before any Phase 2 run:** bidirectional architectures should beat
causal ones on L-NER, because entity type and boundary often depend on
right-hand context, and the gap should be largest for WIT and APP, which are
disambiguated by markers appearing AFTER the name.

| | CV macro-F1 | fold spread | leaderboard |
|---|---|---|---|
| prefix-LM (bidirectional) + CRF | 0.4438 | 0.0165 | 0.5161 |
| **decoder (causal) + CRF** | **0.5044** | **0.0082** | **0.5683** |

**The causal decoder wins by +0.061 CV and +0.052 on the leaderboard**, with
tighter fold agreement.

The per-label detail refutes the mechanism too. Gains are broad rather than
concentrated on role-dependent names: JUDGE +0.191, PREC +0.112, A.COUNSEL
+0.100, R.COUNSEL +0.081, CASENO +0.086 — mostly long structured entities.
WIT actually got worse (-0.114) and RESP slightly worse (-0.026).

**Two explanations, not yet separated:**

1. *The checkpoints are not comparable.* arm_b_docmask (decoder, 3B tokens,
   document masking, Phase 1 clean-set 0.8408) vs arch2_prefix_3B (0.8391).
   Close on Phase 1 but differing in objective, masking and supervision
   density. The NER gain may come from the pretraining difference, not from
   causality.
2. *Causal attention genuinely suits BIO tagging.* BIO is inherently
   left-to-right — I-X depends on what preceded it — so causal representations
   may align better with the tagging structure than bidirectional ones.

**Testable control:** run arm_a_baseline (plain 1.5B decoder, no document
masking) on NER. If it also beats prefix-LM, causality is doing the work; if
it lands near prefix-LM, the gain was the checkpoint. 20 minutes, not yet run.

## 2026-10-10 — Phase 2 status

| # | experiment | CV | leaderboard |
|---|---|---|---|
| 1 | full fine-tuning, prefix-LM | 0.2603 | — |
| 2 | + position features (null) | 0.2578 | — |
| 3 | + CRF head | 0.4438 | 0.5161 |
| 4 | LoRMA+ (compulsory) | 0.4127 | 0.4839 |
| 5 | knowledge distillation (compulsory) | 0.4273 | — |
| 6 | decoder architecture + CRF | **0.5044** | **0.5683** |

Six substantially different experiments, three submissions.

**The full-data gain is now a reliable rule**: retraining on all 70 documents
instead of 35 adds ~0.07 to the leaderboard score, observed three times
(+0.072, +0.071, +0.064). CV therefore systematically underestimates
submission performance here, which is the opposite of the Phase 1 situation
where the local evaluator over-predicted.

**Outstanding:** the arm_a_baseline control; a submission from the distilled
student; and the question of why a causal model wins at span tagging.

## 2026-10-10 — LoRMA on the decoder: the architecture advantage survives PEFT

Running LoRMA+ on arm_b_docmask tests whether the decoder's advantage is a
property of the base model or an artifact of full fine-tuning.

|  | full FT | LoRMA+ | retained |
|---|---|---|---|
| prefix-LM | 0.4438 | 0.4127 | 93.0% |
| **decoder** | **0.5044** | **0.4566** | 90.5% |

The decoder wins under both regimes (+0.061 full, +0.044 LoRMA), so the gap is
not an artifact of the adaptation method. LoRMA retains slightly less on the
better base model — weak evidence that a stronger starting point has more to
lose when only 0.28% of it is adapted, but the difference is small.

Leaderboard 0.5120, against the predicted ~0.52.

Same instability as the prefix-LM LoRMA run: peak at epoch 8-9, then
oscillation between 0.26 and 0.46 while train loss stays flat.

## 2026-10-10 — the 2x2 is complete

| | prefix-LM | decoder |
|---|---|---|
| full FT | 0.5161 | **0.5683** |
| LoRMA+ (0.28% params) | 0.4839 | 0.5120 |

Four leaderboard scores, two architectures x two adaptation regimes, all
tracking CV with a consistent **+0.06 to +0.07 offset** from retraining on 70
documents instead of 35. That offset has now held five times and is reliable
enough to predict submission scores from CV.

Distilled student (6 of 12 layers, decoder teacher): **0.4366**.

Note the discrepancy: CV said the student retained 94.5% of its teacher
(0.4273 vs 0.4520), but on the leaderboard it retains only 77% (0.4366 vs
0.5683). The CV figure came from a prefix-LM teacher and the submission used a
decoder teacher, so the two are not measuring the same thing. The honest
statement is that retention was measured properly only in the prefix-LM
setting.

## 2026-10-10/11 — tuning: nothing left to find

Three attempts to improve on 0.5044 CV for decoder + CRF, all within noise.

**Epochs (10 -> 20): 0.5083.** +0.004, while the fold spread widened from
0.0082 to 0.0217 and the two folds peaked at quite different epochs (11 and 7).
The curves plateau by epoch 3-4 and then wander within a ~0.05 band; the "best"
epoch is wherever the wandering happened to peak. There is no sweet spot to
find.

**Stride (512 -> 256): 0.5050.** +0.0006 on accuracy — no effect. Doubling the
overlap gives every token at least 384 tokens of context on each side instead
of 256, and the model apparently did not need it.

**But the fold spread collapsed to 0.0015**, from 0.0082 at stride 512 and
0.0217 at 20 epochs, with both folds peaking at epoch 8. More windows means
more predictions averaged per document, so less variance from where the
boundaries happen to fall. **Context overlap affects measurement variance, not
accuracy** — worth knowing for anyone reading the fold spreads as evidence.

**Conclusion: the CV noise band here is roughly ±0.025, and nothing tried on
10-11 October exceeded it.** The levers that did work were the CRF (+0.18),
the architecture choice (+0.06), and full-data retraining (+0.06). Decoding
strategy, position features, epochs, and context window were all either null
or noise.

## 2026-10-11 — Phase 2 closed

| # | experiment | CV | leaderboard |
|---|---|---|---|
| 1 | full fine-tuning, prefix-LM | 0.2603 | — |
| 2 | + position features (null) | 0.2578 | — |
| 3 | + CRF head | 0.4438 | 0.5161 |
| 4 | LoRMA+ on prefix-LM (compulsory) | 0.4127 | 0.4839 |
| 5 | knowledge distillation (compulsory) | 0.4273 | 0.4366 |
| 6 | decoder + CRF | **0.5044** | **0.5683** |
| 7 | LoRMA+ on decoder | 0.4566 | 0.5120 |
| 8 | plain head, decoder, tuned decode | — | (baseline submission) |

**Best: 0.5683.** Seven substantially different experiments against the
required six.

### What the phase actually established

- **Sequence structure matters more than representation.** The CRF was worth
  +0.18 — more than every architectural and adaptation choice combined.
- **Parameter-efficient adaptation is close to free.** 0.28% of parameters
  retains 90-94% of full fine-tuning, consistently across two base models.
- **A registered prediction was wrong, and informatively so.** Bidirectional
  attention was predicted to beat causal on span labelling; causal won by
  +0.05 on the leaderboard. Whether that is causality suiting BIO's
  left-to-right structure or simply a better Phase 1 checkpoint remains
  untested — the arm_a_baseline control was descoped for time.
- **Role-dependent entities remain unsolved.** WIT, APP, RESP and R.COUNSEL
  average ~0.11-0.25 F1 across every configuration. Neither architecture,
  adaptation method, decoding strategy, position features, nor document-level
  consistency moved them. The model finds the names and cannot assign the
  roles.

### Open, for the report

- the arm_a_baseline control (20 minutes, resolves why causal won)
- distillation retention measured in the setting it was submitted in
- why `high` has the widest per-label spread across architectures

## Phase 3 — Rhetorical Role prediction

### Setup
Joint CL+IT training, domain marker token (`<|unused1|>`/`<|unused2|>`) prepended
per block; sentences separated by `<|unused0|>`, classification by gathering the
hidden state at each separator ("marker pooling"). Blocks built predicted-region-
first with context back-filled into leftover budget (8 sentences, 1024 tokens),
so no sentence is ever left unpredicted. 80 train / 10 dev docs.

### Task structure
Self-transition rate: 94% CL, 75% IT. This is segment labelling, not independent
sentence classification. Baselines: majority 0.029 macro-F1; per-domain majority
0.029; copy-previous-*gold* oracle 0.754 macro / 0.884 acc. The oracle is the
number to beat the model against — it measures how much of the task is pure
run-continuity.

### Results
| run | dev macro-F1 | acc | CL | IT |
|---|---|---|---|---|
| rr_decoder_plain | 0.5617 | 0.628 | 0.4848 | 0.5845 |
| rr_decoder_crf   | 0.5597 | 0.617 | 0.4744 | 0.5958 |
| rr_prefix_plain  | 0.4518 | 0.542 | 0.4287 | 0.4167 |

At 0.628 accuracy against a 0.884 continuity oracle, the model is capturing
well under half the available run structure.

### CRF: null result (−0.002, inside noise)
Contrast with Phase 2, where the same `crf.py` was worth +0.18. The difference is
the *kind* of structure. In BIO, transitions are hard constraints — `O → I-X` is
not unlikely, it is illegal, and a CRF encodes impossibility exactly. In RR the
labels are merely autocorrelated: every transition is legal, just unevenly
likely. Self-attention over an 8-sentence context already propagates that kind of
soft continuity, so the CRF's transition matrix re-learns something the encoder
has, and buys nothing. Prediction this licenses: a CRF helps a tagger in
proportion to the fraction of its transition matrix that is structurally zero.

### Registered prediction REFUTED (second consecutive phase)
Predicted before running: bidirectional encoding should beat causal here, and by
more than in Phase 2. Observed: it loses by 0.110 (Phase 2: 0.052).
Proposed mechanism: the collapsed labels (Issue 0.100, PrecedentNotReliedUpon
0.244 w/ 87 FP, PrecedentReliedUpon 0.378) are position-dependent rather than
lexically distinctive. A causal pass carries an implicit "depth into the
document" signal; a symmetric prefix erases it. This predicts causal > bidirectional
on any task whose labels are conditioned on document position, and no gap on
tasks whose labels are lexically determined.
Confound, unchanged from Phase 2: arm_b_docmask may simply be the better
pretrained checkpoint. Two downstream tasks do not separate checkpoint quality
from inductive bias.

### Shared failure mode across all three runs
RatioOfTheDecision is massively over-predicted (399 FP). It is the majority label
and the semantic default — anything reasoning-shaped collapses into it.

## Phase 3 — Rhetorical Role prediction

### Setup
Joint CL+IT training with a domain marker token (`<|unused1|>`/`<|unused2|>`)
prepended per block; sentences separated by `<|unused0|>`; classification by
gathering the hidden state at each separator ("marker pooling"). 80 train /
10 dev documents, 1024-token blocks.

### Task structure
Self-transition rate 94% CL / 75% IT — segment labelling, not independent
sentence classification. Baselines: majority 0.029 macro-F1 (acc 0.188);
copy-previous-*gold* oracle 0.754 (acc 0.884). The oracle measures how much of
the task is pure run-continuity and is the number worth comparing against.

Label prior is long-tailed: RatioOfTheDecision 25.5% and Fact 24.3% of training
sentences, PrecedentOverruled 3 sentences in total, Issue 0.64%. Dev contains
**no** Dissent and **no** PrecedentOverruled, so dev macro-F1 averages 11 labels
where the leaderboard uses up to 13.

### All results

| run | dev macro-F1 | acc | leaderboard |
|---|---|---|---|
| decoder, context=0 (`rr_decoder_plain`) | 0.5617 | 0.628 | 0.5023 |
| decoder + CRF | 0.5597 | 0.617 | — |
| decoder + class weights (inv_sqrt) | 0.5478 | 0.609 | — |
| prefix-LM, context=0 | 0.4518 | 0.542 | — |
| LoRMA, context=0 | 0.5248 | 0.583 | — |
| decoder, all 90 docs, no early stopping | — | — | 0.4626 |
| decoder, context=8 | 0.5720 | 0.629 | — |
| decoder, context=24 | 0.5894 | 0.652 | 0.5014 |
| context=24, test blocks built at context=8 (bug) | — | — | 0.4919 |
| distilled student (6/12 layers) | 0.5234 | 0.592 | 0.4503 |

### The methodological finding (the important one)
The distillation log gives a direct estimate of dev noise. Within a single run,
with training loss falling monotonically, dev macro-F1 over epochs 5–12 was
0.5065 / 0.4592 / 0.4900 / 0.5228 / 0.4826 / 0.5234 / 0.4891 / 0.4825 —
**mean 0.495, sd ≈ 0.021**, adjacent epochs differing by up to 0.06.

Every run in this phase reports its best epoch. The maximum of 12 draws at
sd 0.021 around a flat mean is inflated by roughly 1.5 sd ≈ **+0.03**. That one
fact accounts for the whole dev→leaderboard pattern:

  ctx0   dev 0.5617 (best-of-12) -> ~0.53 de-biased -> 0.5023 board
  ctx24  dev 0.5894 (best-of-12) -> ~0.56 de-biased -> 0.5014 board

and it means every effect chased this phase (0.01–0.03) was at or below the
measurement noise of the instrument used to chase it. 10 documents / 2142
sentences cannot rank models at this resolution.

What DOES survive: ratios measured on the same split. Distillation retention
was 88.8% on dev and 89.8% on the leaderboard.

### Six interventions that re-slice the same posterior: all null
| intervention | effect |
|---|---|
| CRF over the label sequence | −0.0020 dev |
| class weighting (inv_sqrt) | −0.0139 dev |
| logit adjustment, tau 0–1.0 | flat within 0.008 across the sweep |
| 3-model majority vote | +0.0060 dev |
| 4-model majority vote | +0.0072 dev |
| training on all 90 documents | **−0.0397 leaderboard** |

CRF contrast with Phase 2, where the same `crf.py` was worth +0.18: BIO has
*hard* structure (`O -> I-X` is illegal, not merely unlikely) and a CRF encodes
impossibility exactly. RR labels are only autocorrelated — every transition is
legal — and self-attention already propagates that. Conjecture: a CRF helps in
proportion to the fraction of its transition matrix that is structurally zero.

Logit adjustment null is informative: if Ratio were merely over-confident,
shifting the decision threshold would trade its precision for minority recall and
macro-F1 would rise. Ratio's predicted share fell 28.0% -> 25.6% across the sweep
with macro-F1 flat. The Ratio/ArgumentPetitioner confusion is a representation
failure, not a calibration failure.

All-data negative result inverts the standard "retrain on everything before
submitting" move: with 90 documents memorisable (110M params reach 0.996 dev-on-
train macro-F1 by epoch 9, train loss 0.002), the stopping point is the most
important hyperparameter, and `--all_data` deletes the signal that sets it.
**The 10 held-out documents are worth more as a stopping criterion than as 12.5%
more training data.**

### Context: the one lever that moved anything
A latent bug meant `context` was ~0 in every run up to this point. The
three-pass block builder filled the predicted region first (`if budget <= 0:
break`), leaving nothing for lead-in back-fill. Fixed by reserving the context
budget before filling the predicted region, capped at `ctx_frac` of the block.
A second latent bug: pass 3 emitted a marker for every sentence including
context, which was harmless only while context was 0 — with context working it
would have shifted every label. Now asserted (`len(markers) == end - start`).

| context | mean lead-in sents | blocks | dev macro-F1 | acc |
|---|---|---|---|---|
| 0 | 0.00 | 736 | 0.5617 | 0.6284 |
| 8 | 6.36 | 976 | 0.5720 | 0.6289 |
| 24 (ctx_frac 0.5) | ~11.5 | 1321 | 0.5894 | 0.6522 |

Monotone, and the second doubling bought MORE than the first (+0.017 vs +0.010),
refuting the registered prediction that most disambiguating signal sits in the
immediately preceding sentences. Per-label, the gain concentrates exactly where
predicted: Issue 0.500 -> 0.714, ArgumentPetitioner 0.495 -> 0.631 (false
negatives 242 -> 188), RatioOfTheDecision false positives 334 -> 263.
Position-dependent labels benefit; lexically-determined ones (Statute) do not.

Caveat, given the noise analysis: +0.028 total is the size of the measurement
noise. Monotonicity across three points is weak supporting evidence, and the
leaderboard move was +0.000 (0.5023 -> 0.5014). NOT established.

Hard ceiling: 1024-token blocks, because Phase 1 used learned absolute position
embeddings. At ctx_frac 0.65 only a third of each block is predicted, so compute
triples for shrinking returns. Going further requires a hierarchical encoder —
sentence vectors from the Phase 1 encoder, then a second small transformer over
all sentences of a document — not reachable by tuning.

### Registered predictions
1. "Bidirectional should beat causal here, and by more than in Phase 2."
   **REFUTED**, second consecutive phase. Prefix-LM lost by 0.110 (Phase 2:
   0.052). Proposed mechanism: collapsed labels (Issue 0.100,
   PrecedentNotReliedUpon 0.244 with 87 FP, PrecedentReliedUpon 0.378) are
   position-dependent rather than lexically distinctive; a causal pass carries an
   implicit "depth into the document" signal that a symmetric prefix erases.
   Confound unchanged from Phase 2: `arm_b_docmask` may simply be the better
   checkpoint; two downstream tasks do not separate checkpoint quality from
   inductive bias.
2. "More context helps transition-marking labels disproportionately."
   **CONFIRMED** directionally (Issue +0.21, ArgumentPetitioner +0.14), magnitude
   over-predicted (+0.010 observed vs +0.02–0.04 predicted).
3. "The second context doubling buys less than the first." **REFUTED** — it
   bought +0.017 vs +0.010.

### LoRMA (compulsory) — dev 0.5248 @ epoch 5
294,912 adapter params over 12 Q/V projections (r=8, alpha=16) plus head =
304,909 / 110,621,197 = **0.28% trainable**. max_lr 3e-4; the 3e-5 full-finetune
rate badly underfits adapters. Retains 93.4% of full-finetune macro-F1.
Phase 2 NER retention was 90–95% at 0.284% — near-identical ratios on tasks with
different output structure, suggesting the retention rate is a property of
multiplicative low-rank adaptation rather than of the task.

Capacity corollary: full finetune (110M trainable) and LoRMA (0.3M trainable)
land 0.037 apart and both saturate by epoch 5–10. Parameter count is not the
binding constraint; 80 documents is.

### Knowledge distillation (compulsory)
Teacher `rr_ctx24` (110,326,285 params, 12 layers, dev 0.5894). Student = every
other layer (6/12, DistilBERT scheme), embeddings/head copied, 67,849,741 params
(61.5%), 2.05x faster. Two arms from ONE initialisation, identical schedules.

  teacher            dev 0.5894     board 0.5014
  student, distilled dev 0.5234     board 0.4503
  student, hard only dev 0.5184     board —
  **distillation effect +0.0050**

Phase 2 NER measured +0.0036 with an emissions-only KL (CRF teacher). Here the
teacher has no CRF so the KL is exact, and the effect replicates. Two tasks,
different output structure, approximate vs exact KL — same near-zero answer.

Claim: the value in distillation here is the layer-selection initialisation and
the student's capacity, not the teacher's soft distribution. The student learns
from hard labels; matching the teacher's output adds nothing measurable.

Both effects (+0.005, +0.0036) are below the sd 0.021 noise floor, so each alone
is a null. The replication across two tasks is what makes it a claim.

### Bugs that cost submissions
- `--all_data` reuse of a dev-selected epoch count: −0.040 leaderboard.
- Test blocks built with default context=8 against a context=24 checkpoint:
  −0.0095 leaderboard (0.5014 -> 0.4919). Symptom was RatioOfTheDecision at
  43.7% of test sentences vs 32% on dev for the same model.
  Fixed by reading block geometry from `ck["args"]` and printing it, and by
  recording it in `config.txt`.
- Guard adopted late: compare a submission's predicted label distribution against
  the same model's dev distribution before uploading. A >10-point divergence in
  the majority class indicates a train/test geometry mismatch.

## Phase 4 — Court Judgment Prediction (CJPE)

### Data
train 37,387 cases (59/41) | dev 1,988 (50/50) | test 1,517 (50/50).
Token lengths: train median 2,669, dev 3,800, test 3,938 against a 1024-token
ceiling (Phase 1 learned absolute positions). Only 20% of training cases and
3.8% of dev cases fit in one window. Document length does NOT leak the label
(class medians within 1%). The `expert_1..5` fields are empty in all splits.

Two distribution shifts between train and evaluation:
  1. class prior 59/41 -> 50/50
  2. evaluation documents are ~30% longer than training ones

Dev at 1,988 balanced cases gives binomial sd ~0.011 on accuracy, so effects
above ~0.02 are measurable — a different regime from Phase 3, where 10
documents made best-epoch selection alone worth ~+0.03 of illusory gain.

NOTE: `test.json` ships a populated `label` field although the brief states test
labels are private. Not read: `cjpe_data.load_split` drops the key on the test
split and `make_cjpe_submission` asserts its absence. Should be reported to the
instructors.

### Representation: where in a judgment is the outcome?
One 1024-token window per case, last-token pooling, 3 epochs, decoder
(arm_b_docmask), identical otherwise.

| window | dev macro-F1 |
|---|---|
| tail (last 1024) | **0.7124** |
| head_tail (first 512 + last 512) | 0.6875 |
| head (first 1024) | 0.5429 |

Registered prediction (head loses to tail by > 0.03) CONFIRMED at 0.170.

The head arm did not merely score lower, it never learned: train loss stalled at
0.64 through epoch 1 where tail was at 0.48, then collapsed to predicting class 1
on 98% of dev (macro-F1 0.354, essentially the 0.3333 majority floor), while
train loss fell to 0.37 by epoch 3 — memorisation without generalisation. The
opening of a judgment (facts, procedural history) carries no outcome signal.

head_tail is the informative number: spending half the budget on the head COSTS
0.025. Head tokens are not merely weaker, they are worth less than the tail
tokens they displace. Consequence for any hierarchical design: uniform pooling
over chunks is wrong, since it would dilute the one chunk that matters.

### Prior correction — the Phase 3 null explained
Evaluation reports macro-F1 both raw and after subtracting tau*log p_train(y).
At the selected checkpoint the correction is worth **+0.023** (0.6895 -> 0.7124).
Across the run it tracks the model's bias exactly: +0.077 at step 1500 when
pred1 = 0.182, shrinking as pred1 approaches 0.5, and NEGATIVE at step 13500
when pred1 = 0.588. It corrects a real bias rather than adding a constant.

Phase 3's logit-adjustment sweep was flat (0.000) across tau 0–1.0. The
difference is that Phase 3's train and dev shared a prior, so there was nothing
to correct. Same intervention, opposite outcome, mechanism confirmed.

### Adaptation techniques (all --mode tail)

| technique | Phase 1 model | pooling | trainable | dev macro-F1 |
|---|---|---|---|---|
| full fine-tuning | decoder (arm_b_docmask) | last | 100% | **0.7124** |
| knowledge distillation | decoder student 6/12 | last | 61.5% | 0.6911 |
| LoRMA r=8 a=16 Q/V | prefix-LM | mean | 0.27% | 0.6836 |
| full fine-tuning | encoder-decoder | mean | 100% | 0.6342 |
| head-only probe (see below) | efficient | last | 1,538 | 0.5321 |

LoRMA retains **96%** of full fine-tuning at 0.27% of parameters — third
consecutive phase in the 90–96% band (Phase 2: 90–95%, Phase 3: 93.4%) across
three tasks with different output structure. The retention rate looks like a
property of multiplicative low-rank adaptation rather than of the task.

CONFOUND, stated: architecture and pooling are entangled. The decoder used
last-token pooling, prefix-LM and encdec used mean. "encdec scores 0.6342" may
be the architecture or the pooling; one decoder run with --pooling mean would
deconfound the table. Not run.

### BitFit is inapplicable to this architecture
The BitFit arm trained 1,538 parameters = 768*2 + 2, the classification head
alone. Zero base parameters matched `.bias`: the Phase 1 models were built
nanoGPT-style with `Linear(bias=False)` and RMSNorm (scale only), so there are
no bias terms to tune. BitFit cannot be run on these models at all.

The arm is still informative as a head-only linear probe: a frozen encoder
reaches 0.5321, so roughly 0.18 of the final 0.7124 comes from ADAPTING the
encoder rather than from the pretrained representation.

### Knowledge distillation (compulsory) — effect +0.0354
Teacher cjpe_tail (110,317,826 params, dev 0.7124). Student = every other layer
(6/12), 67,841,282 params (61.5%). Two arms from one initialisation, 2 epochs.
  distilled 0.6911 | hard only 0.6557 | **effect +0.0354** | retention 97.0%
At ~2.2 sd on a 1,988-case dev, this is a real effect — unlike the sub-noise
nulls of Phases 2 and 3.

REGISTERED PREDICTION REFUTED. Predicted the KL effect would be SMALLER than
Phase 2/3's ~+0.004, on the grounds that a 2-class soft target is nearly fully
described by its hard label. Observed ~7x LARGER.

Revised account: the KL term is a regulariser whose value scales with
training-set size, not with the entropy of the label space.
  70 docs -> +0.0036 | 80 docs -> +0.0050 | 37,387 cases -> +0.0354
With 70–80 documents the student memorises the hard labels within a few epochs
and no softening changes the solution. With 37k examples at 2 epochs the student
is underfitting, and the teacher's probabilities encode which cases are
borderline — the teacher is only 71% accurate, so a large fraction are.
Confounds: task, class count and epoch budget also differ. The clean test is a
data-size sweep on CJPE alone (--limit_train 2000 / 8000); not run.

### Submissions
| | model | dev |
|---|---|---|
| p4_sub1 | full fine-tune, tail 1024, prior-corrected | 0.7124 |
| p4_sub2 | head+tail 512/512 | 0.6875 |
| p4_sub3 | head 1024 (position ablation) | 0.5429 |
| p4_sub4 | LoRMA on prefix-LM | 0.6836 |
| p4_sub5 | full fine-tune on encoder-decoder | 0.6342 |
| p4_sub6 | distilled student 6/12 layers | 0.6911 |

Submission script reads mode / pooling / max_len / decision-rule from the
checkpoint rather than the command line, and installs LoRMA adapters or
re-truncates a distilled student before `load_state_dict` — the generalised fix
for the Phase 3 bug where test windows were built with different geometry than
training and cost 0.0095 on the leaderboard.

### Not done (brief section 10.1)
Multi-chunk / sliding-window / hierarchical document representations were not
implemented; every run uses a single 1024-token window chosen by position. The
position ablation gives the hypothesis a hierarchical model would test: a
learned attention over the last N chunks should concentrate on the final chunk,
and gain little over tail-1024, since head_tail showed earlier material actively
displaces better material.

## Phase 5 — Legal Statute Identification (LSI, multi-label)

### Data
train 42,750 | dev 10,181 | test 13,019 documents. 100 statutes, ids 0-99, all
present in both labelled splits, every statute appearing >20 times — so unlike
Phase 3 there are no unreachable labels and macro-F1 has no structural ceiling.
Mean 3.71 labels/doc (train), 3.92 (dev); 21% of training docs have exactly one.
Head 10 statutes cover 41% of all label occurrences; most common appears in 27%
of documents.

Documents are SHORT relative to CJPE: median 643 (train) / 804 (dev) tokens,
61% / 56% fitting in a 1024-token window. Dev at 10,181 documents gives noise
sd ~0.005, the most trustworthy evaluation set of any phase.

### Headline: this task is a calibration problem
Full fine-tune, decoder, head-1024, mean pooling, identical model, three
decision rules:

| rule | dev macro-F1 |
|---|---|
| fixed 0.5 | 0.2457 |
| best global threshold (0.15) | 0.3572 |
| per-label thresholds | 0.3849 |

**Threshold choice is worth +0.139.** The entire spread across four adaptation
techniques and four architectures is 0.049. Converting scores to labels matters
roughly three times more than how the model is adapted.

Registered prediction (per-label thresholding beats the full-FT-vs-LoRMA gap)
CONFIRMED: +0.139 vs 0.029 in Phase 4, and vs 0.031 here.

Mechanism: the optimal global threshold is 0.15, not 0.5. A statute positive in
a small share of documents rarely reaches p=0.5 even when ranked correctly, so a
0.5 cutoff systematically drops the tail — which is exactly what macro-F1
averages over. Most of the gain is the global shift; per-label tuning adds a
further +0.028 of genuine per-statute variation.

### Adaptation techniques (all --mode head --pooling mean)

| technique | Phase 1 model | trainable | dev macro-F1 |
|---|---|---|---|
| full fine-tune, 4 epochs | decoder | 100% | 0.4004 |
| top-4-layer fine-tune | efficient | 25.7% | 0.4004 |
| LoRMA r=8 Q/V | prefix-LM | 0.34% | 0.3689 |
| full fine-tune | encoder-decoder | 100% | 0.3512 |
| full fine-tune, 2 epochs | decoder | 100% | 0.3849 |

Top-4-layer fine-tuning exactly matches full fine-tuning at 25.7% of
parameters: adapting the top third loses nothing. 2 -> 4 epochs is worth
+0.0155, so the first run was undertrained rather than converged.

### Knowledge distillation (compulsory) — student exceeds teacher
Teacher lsi_full3 (110,393,188 params, dev 0.4004). Student every-other-layer
(6/12), 67,916,644 (62%). Soft targets are the teacher's per-statute sigmoid
probabilities matched with BCE at T=2.
  distilled 0.4030 | hard only 0.3947 | **effect +0.0083** | retention 100.7%

REGISTERED PREDICTION REFUTED (second distillation prediction in a row).
Predicted "at least as large as CJPE's +0.0354" on the grounds that 100 graded
targets per document is denser supervision than CJPE's one. Observed +0.0083.

Four measurements now:
  Ph2 NER   70 docs    teacher 0.4438                         effect +0.0036
  Ph3 RR    80 docs    teacher 0.5894  hard-only 0.5184       effect +0.0050
  Ph4 CJPE  37,387     teacher 0.7124  hard-only 0.6557       effect +0.0354
  Ph5 LSI   42,750     teacher 0.4004  hard-only 0.3947       effect +0.0083

The Phase 4 account (effect scales with training-set size) is dead: Phases 4 and
5 have comparable training sizes and differ 4x. Current hypothesis, stated as
hypothesis: distillation helps in proportion to how much the truncation actually
COSTS, but only when there is enough data to exploit softer targets. CJPE's
student loses 0.057 to its teacher on hard labels and KD recovers 62% of that;
LSI's student loses only 0.0057, so six layers already suffice and KD acts as a
mild regulariser that pushes the student PAST the teacher. Phase 3 had the gap
(0.071) but not the data, and recovered almost nothing. Untested prediction: a
--limit_train sweep on CJPE should shrink the effect toward Phase 3's.

### Long-document representation (brief 11.1)
Chunked inference at stride 768, up to 8 chunks, on the trained single-window
model — no retraining. 10,181 dev docs -> 24,793 chunks (2.44/doc).

| aggregator | @0.5 | best global | per-label |
|---|---|---|---|
| max | 0.3006 | 0.3836 | **0.4114** |
| mean | 0.2318 | 0.3844 | 0.4108 |
| noisy-or | 0.3339 | 0.3708 | 0.3955 |

Registered prediction REFUTED: predicted max would clearly beat mean because
statutory evidence is local and averaging dilutes a confident local detection.
At per-label thresholds they are TIED (0.4114 vs 0.4108, a tenth of the noise
floor). At a fixed 0.5 cutoff max looks dramatically better (0.3006 vs 0.2318).
**The apparent superiority of max-pooling is entirely a calibration artifact** —
the aggregators preserve the same ranking and differ only in output scale.
A comparison of pooling strategies at a default 0.5 cutoff would report a 0.07
improvement that does not exist.

Noisy-or is worst because it saturates with chunk count, giving 8-chunk
documents systematically higher scores than 1-chunk ones — a per-document bias
that no per-label threshold can correct.

Reading the whole document instead of the first 1024 tokens is worth only
+0.0084, although 44% of dev documents were being truncated. Statutory evidence
is concentrated EARLY, consistent with facts and charging sections appearing
near the start. Mirror image of Phase 4, where CJPE's signal lived entirely at
the tail (head 0.5429 vs tail 0.7124): same corpus family, opposite position
structure, because the two tasks ask different questions of the document.

### Cross-architecture ensemble — Phase 3's explanation half-confirmed
Five members over four Phase 1 architectures and four adaptation techniques,
probabilities averaged, per-label thresholds retuned: **0.4132**, +0.0102 over
the best single member (0.4030).

Pairwise correlation of member probabilities:
  same pretrained checkpoint (lsi_distil / lsi_full3)   0.972
  every cross-architecture pair                         0.865 - 0.895

Phase 3's ensemble gained +0.007 and I attributed the failure to members sharing
one pretrained checkpoint and 80 documents. That mechanism is REAL — different
Phase 1 pretraining measurably decorrelates predictions, 0.97 -> 0.87 — but the
payoff did not follow: +0.0102 against Phase 3's +0.007, with 500x the data.

Conclusion: the residual errors are TASK-INTRINSIC, not model-specific. Four
architectures with different attention patterns and pretraining objectives,
trained on the same legal corpus, converge on the same hard documents and
ambiguous statutes. Model diversity does not create diversity in what is
learnable from the data.

Third instance this phase of a fixed threshold inverting a conclusion: the
ensemble at 0.5 scores 0.2352, WORSE than every member, because averaging
shrinks probabilities toward the middle. Only per-label retuning recovers the
gain.

### Submissions
| | model | dev |
|---|---|---|
| p5_sub1 | full fine-tune decoder, 4 epochs | 0.4004 |
| p5_sub2 | top-4-layer fine-tune, efficient arch | 0.4004 |
| p5_sub3 | LoRMA on prefix-LM | 0.3689 |
| p5_sub4 | full fine-tune encoder-decoder | 0.3512 |
| p5_sub5 | distilled student 6/12 layers | 0.4030 |
| p5_sub6 | chunked inference, max aggregation | 0.4114 |
| p5_sub7 | 5-model cross-architecture ensemble | **0.4132** |

### Not done
pos_weight / loss-function variants (the one lever in brief 11.1 untouched);
chunked inference stacked on the ensemble; label co-occurrence modelling.

### Cross-phase note
Phases 3, 4 and 5 all hinged on a decision rule rather than a representation:
Phase 3's logit adjustment was null because train and dev shared a prior;
Phase 4's prior correction was worth +0.023 because they did not; Phase 5's
per-label thresholds were worth +0.139 because macro-F1 over a long-tailed label
set is dominated by labels a default cutoff never fires on. The same family of
intervention ranged from 0.000 to +0.139 depending entirely on whether a
mismatch existed between the training distribution and the evaluation metric.

## Phase 6 — Abstractive Legal Summarization

### Data
7,030 train examples (6,528 after dropping empty summaries), 100 test documents.
No public dev split: 500 documents held out from train by a seeded shuffle
(SPLIT_SEED 1899) in summ_data.load_split, so train/dev are a deterministic
partition and every run sees the same split.

Documents median 3,656 tokens, 97% over the 1024 ceiling. Summaries median 749
tokens, mean 991 — a summary ALONE nearly fills a 1024-token window, so there is
no format in which document and summary share one context. Compression 4.6x.
Reference summaries share 86.1% of their token types with the source document.

### The metric reads only the first 512 tokens
BERTScore uses roberta-large, whose tokenizer caps at 512. Confirmed two ways:
`s._tokenizer.model_max_length == 512`, and appending 500 junk tokens past the
cutoff leaves the score bit-identical. Lead-k extraction on dev (200 docs):

| output | BERTScore F1 |
|---|---|
| lead-128 | 0.7981 |
| lead-256 | 0.8173 |
| lead-384 | 0.8263 |
| lead-512 | **0.8300** |
| lead-750 / lead-1000 / whole document | 0.8300 (identical) |

Consequences, all acted on:
  - generation length has a hard optimum: fill ~512 tokens, then stop. Worth
    +0.032 from 256 to 512 and exactly 0 beyond.
  - only the OPENING of each reference is ever scored, so training uses chunk 0
    (source = first 1024 document tokens, target = first 512 summary tokens).
    The chunk-pair machinery is retained for coverage/ROUGE and labelled as a
    different experiment.
  - the brief's hierarchical-summarization emphasis serves the SECONDARY metrics
    (ROUGE sees everything); it cannot move the primary one.

### The baseline that matters
**lead-512 extraction = 0.8300, and no abstractive model built here beat it.**
Best trained model: 0.8290. With 86% token overlap and a 512-token embedding
comparison, BERTScore on this corpus largely measures "is this legal text about
the same case", which a verbatim prefix satisfies by construction. Reported as
the headline rather than a footnote.

### Decoding dominates, and a fixed decoder inverts architecture rankings
Same two checkpoints, swept over six decoding configurations:

| model | greedy-480 | best config | best |
|---|---|---|---|
| prefix-LM + LoRMA | 0.7894 (rep4 0.806) | greedy-480-norep4 | **0.8290** |
| encdec full FT | 0.8015 (rep4 0.477) | topk50-480 | 0.8212 |
| decoder top-4 | 0.7973 (rep4 0.540) | greedy-256 | 0.8084 |

Under the training loop's fixed greedy-480, the encoder-decoder appears to beat
prefix-LM by 0.012. Under each model's own best decoding, prefix-LM wins by
0.008 — **the ranking inverts**. Prefix-LM gains +0.0396 from one no-repeat-4
flag because greedy decoding sends it into a 0.806-repetition loop.

Suppressing EOS to fill the metric window CAUSES the looping: the model has said
what it knows and cannot stop. Length and non-repetition are both required, and
neither alone suffices — greedy-free stops at 196-297 tokens and loses recall;
greedy-480 fills the window with loops and loses precision. The combination is
worth more than three epochs of training.

Earlier speculation that BERTScore might REWARD on-topic repetition is refuted:
at identical length (480), blocking 4-grams gains +0.024.

Each model's optimum differs — greedy+norep for prefix-LM, top-k sampling for
encdec, and greedy-256 for the decoder, which alone prefers to stop EARLY
because its content degrades with length.

This is the Phase 5 lesson in a different costume. There, a fixed 0.5 threshold
made max-pooling look far better than mean (pure artifact) and made the ensemble
score worse than its members. **Any comparison made under a fixed decision rule
measures the decision rule as much as the thing compared**, and in both phases
the default choice produced the wrong answer.

### Registered prediction CONFIRMED (after two failures)
Predicted prefix-LM would win this phase, summarization being its native shape
(bidirectional over the source, causal over the summary) rather than an
adaptation of it — after predicting prefix-LM wins in Phase 2 (lost by 0.052)
and Phase 3 (lost by 0.110). It wins at 0.8290 vs 0.8212, AND at 0.34% trainable
parameters against a fully fine-tuned encoder-decoder.
Caveat: confounded with adaptation technique (LoRMA vs full), so this does not
cleanly settle the checkpoint-quality-vs-inductive-bias question from Phases 2-3.
Note it only wins under matched decoding; the fixed-decoder comparison hid it.

### Trainable fraction vs overfitting
6,528 examples is small for generation, and the val-loss trajectories order by
parameter budget rather than by architecture quality:

| run | trainable | trajectory |
|---|---|---|
| summ_lorma (prefix-LM) | 0.34% | still improving at ep 5, flat at 6 |
| summ_encdec6 | 100% | val 2.88 -> 3.05, mild overfit |
| summ_topk (decoder) | 25.7% | val 2.70 -> 2.99, peaked ep 3, severe |

LoRMA's budget acts as a regulariser; its lower absolute score reflects
undertraining rather than being outclassed. Val losses are NOT comparable across
architectures (different loss formulations) — only the trajectories are.

### Knowledge distillation (compulsory) — effect +0.0004
Teacher summ_lorma (110,611,200 params, 0.8290 on matched decoding). Student
6/12 layers, 67,987,200 (61.5%), trained fully. Both arms evaluated under
greedy+norep4, not the training-loop default — the distinction that inverted the
architecture ranking above.
  teacher 0.8290 | distilled 0.8273 | hard only 0.8269 | **effect +0.0004**

**Halving the depth costs only 0.0017**, so there was nothing for distillation to
recover. My registered prediction (a LARGER effect than LSI's, since generation
is a harder target and the student should lose more) was wrong in its premise:
the student lost almost nothing.

But the two-factor account formed after Phase 4 now fits all four phases without
modification — the KD term helps in proportion to what truncation COSTS, and
only when there is enough data to exploit soft targets:

| phase | truncation cost | train size | KD effect |
|---|---|---|---|
| Ph3 RR | 0.0710 | 80 | +0.0050 |
| Ph4 CJPE | 0.0567 | 37,387 | **+0.0354** |
| Ph5 LSI | 0.0057 | 42,750 | +0.0083 |
| Ph6 Summ | 0.0017 | 6,528 | +0.0004 |

Phase 4 is the only phase with both a large gap and large data, and the only one
with a large effect. Phase 3 had the gap without data; Phases 5 and 6 had data
without a gap. The account was formed after Phase 4 and has survived two
subsequent tests unchanged.

Secondary observation: the hard-only arm peaked at epoch 4 and declined
(0.8269 -> 0.8225) while the distilled arm improved monotonically to epoch 5.
Even where KD adds nothing to the final score it suppresses overfitting —
consistent with the LoRMA regularisation pattern above.

### Submissions
| | model | technique | arch | dev BERTScore |
|---|---|---|---|---|
| p6_sub1 | summ_encdec (3 ep) | full fine-tuning | encdec | 0.8225 |
| p6_sub2 | summ_lorma | LoRMA (0.34%) | prefix-LM | **0.8290** |
| p6_sub3 | summ_topk | partial/layer-wise (25.7%) | decoder | 0.8084 |
| p6_sub4 | summ_distil | knowledge distillation | prefix-LM 6/12 | 0.8273 |
(lead-512 extractive control, not submitted: 0.8300)

### Not done
Hierarchical / multi-chunk summarization with document-level aggregation (the
brief's main long-document section) — deliberately, since it cannot affect a
metric that reads 512 tokens, though it would serve the ROUGE secondaries.
ROUGE-1/L were never measured locally; only BERTScore was optimised.

### Cross-phase closing note
Phases 3-6 each turned on a decision rule rather than a representation:
Phase 3's logit adjustment null (no prior shift to correct), Phase 4's prior
correction +0.023 (shift present), Phase 5's per-label thresholds +0.139
(long-tailed labels a default cutoff never fires on), Phase 6's decoding
configuration +0.040 and an inverted architecture ranking. The same family of
intervention ranged from 0.000 to +0.139, and in three of the four phases the
DEFAULT decision rule would have produced a materially wrong conclusion about
which model or method was better.

## Phase 7 — Prior Case Retrieval

### Data
train 827 queries / 4,320 candidates | dev 118 / 1,023 | test 237 / 1,727.
Each split has its own pool and they are never mixed. Pools overlap across
splits (42.6% of the dev pool also appears in train), so document-identity
features can transfer to dev; not leakage, since relevance is query-specific,
but a reason not to over-trust dev.

Relevant candidates per query: train mean 6.42 (median 4), dev mean 8.25
(median 5), max 92. Exactly 7 gold: 5-8% of queries; fewer than 7: ~60%.

Documents are the longest of any phase: query median 5,789 tokens, candidates
median 4,680 with a maximum of 514,875.

### The ceiling is 0.6181, not 1.0
Exactly 7 predictions are required and only the top 7 are scored, while 60% of
queries have fewer than 7 gold candidates and some have up to 92. With PERFECT
ranking on dev: micro-P 0.6731, micro-R 0.5714, **micro-F1 0.6181**. Random-7
scores 0.0068. Every result below is read against those two numbers.

### BM25 control: truncation is a feature, not a compromise
| budget | F1@7 | | budget | F1@7 |
|---|---|---|---|---|
| 256 | 0.1412 | | 4096 | **0.2991** |
| 512 | 0.2023 | | 8192 | 0.2357 |
| 1024 | 0.2468 | | 16384 | 0.1912 |
| 2048 | 0.2879 | | 32768 | 0.1434 |
| | | | full | 0.0734 |

Reading the WHOLE document makes BM25 4x worse than reading the first 4,096
tokens. Registered prediction (BM25 over full documents beats dense) REFUTED:
full-document BM25 is the worst method tested.

Length normalisation was the obvious explanation and it is wrong: full documents
at b=0.0 score **0.0111**, four times worse than b=0.75's 0.0734. So
normalisation is doing real work (without it the 514k-token candidates match
everything) but cannot distinguish signal tokens from boilerplate ones.

Account: discriminative content sits in roughly the first 4,096 tokens
(headnote, facts, issues). Beyond that, Indian judgments are dominated by shared
boilerplate — statutory citations, procedural formulae, quoted precedent — which
inflates spurious term overlap between unrelated cases. The optimum sits at
exactly 4x the model's 1024-token window, which set the dense design.

### Representation beats everything else tested
| | dev F1@7 |
|---|---|
| random | 0.0068 |
| zero-shot dense, single 1024 window | 0.1946 |
| zero-shot dense, 4-chunk max-sim | **0.2557** |
| BM25 @4096 | 0.2991 |
| trained bi-encoder (in-batch negatives) | **0.3346** |
| perfect ranking | 0.6181 |

**Chunked max-sim is worth +0.061 with no training at all** — representing a
case as up to 4 chunks over the first 4,096 tokens and scoring a pair by the
best-matching chunk pair, rather than averaging one window. Max rather than
mean for the Phase 5 reason: relevance between two judgments often turns on one
matching passage, and averaging dilutes it. Contrastive training then adds
+0.079 on top, clearing BM25 by +0.036.

Ordering of what mattered: representation (+0.061) > training (+0.079 but it
needs the representation) >> technique spread (0.018) ~ objective (0.010).

### Objective: hard negatives do NOT help
In-batch negatives 0.3346 | BM25-mined hard negatives 0.3246.
Registered prediction (hard negatives beat in-batch by more than the technique
spread) REFUTED on both halves — hard negatives are 0.010 WORSE, and 0.010 is
smaller than the 0.018 technique spread. At 118 dev queries (sd ~0.015) the
honest claim is "no benefit", not "harmful".

Likely mechanism, testable and not tested: PCR gold sets are badly incomplete —
8.25 labelled relevant candidates out of 1,023, with many genuinely relevant
prior cases unlabelled. BM25's top-20 non-gold candidates are precisely the ones
most likely to be unlabelled positives, so the model is trained to push away
true matches. Prediction: sampling negatives from ranks 50-200 instead of 1-20
recovers the loss.

This also bounds the Phase 5/6 pattern ("the decision rule matters more than the
technique"). It does NOT extend to the training objective here.

### Adaptation techniques
| technique | arch | trainable | zero-shot | trained |
|---|---|---|---|---|
| full fine-tuning | decoder | 100% | 0.2557 | **0.3346** |
| LoRMA r=8 Q/V | prefix-LM | 0.27% | 0.1723 | 0.3202 |
| top-4-layer | efficient | 25.67% | 0.2535 | 0.3168 |
| distillation (6/12) | decoder | 61.5% | — | 0.3157 |

All four clear BM25's 0.2991. Prefix-LM transfers WORST zero-shot (0.1723 vs
0.2557) yet closes most of the gap at 0.27% trainable — its pretrained
representation is poorly aligned to similarity search but readily adaptable.
pcr_topk was still improving at epoch 6 (monotone throughout), so it is
undertrained and its number is a floor.

### Knowledge distillation (compulsory) — effect +0.0133
Teacher pcr_full (110,316,288 params, 0.3346). Student 6/12 layers,
67,839,744 (61.5%). Retrieval-specific signal: the student matches the teacher's
SIMILARITY DISTRIBUTION over each batch (KL on softmax(q.c / T)), i.e. the
ranking signal, alongside its own InfoNCE against the gold positive.
  teacher 0.3346 | distilled 0.3157 | hard only 0.3024
  **effect +0.0133 | truncation cost 0.0322**

REGISTERED PREDICTION REFUTED (narrowly): predicted under +0.01 regardless of
truncation cost, on the grounds that 5,308 training pairs is too little data to
exploit a soft target.

Five-phase series, with the two factors proposed after Phase 4:
| phase | truncation cost | train size | KD effect | effect/cost |
|---|---|---|---|---|
| Ph3 RR | 0.0710 | 80 docs | +0.0050 | 0.07 |
| Ph4 CJPE | 0.0567 | 37,387 | +0.0354 | 0.62 |
| Ph5 LSI | 0.0057 | 42,750 | +0.0083 | 1.46 |
| Ph6 Summ | 0.0017 | 6,528 | +0.0004 | 0.24 |
| Ph7 PCR | 0.0322 | 5,308 pairs | +0.0133 | 0.41 |

Honest status: the account gets the ORDERING roughly right (the two large-data
phases have the highest recovery ratios, the 80-document phase the lowest) but
there is no clean functional form over five points, and it mispredicted here.
Reported as a working account that was right twice and wrong once, not as
established.
Caveat: both arms were still improving at epoch 6, so this comparison is taken
at a non-converged point.

Practical note: the distilled student is the most useful model of the four for
retrieval, where the whole candidate pool must be embedded and stored — 94.4%
of the teacher's F1 at 61.5% of the parameters. The hard-only arm at 0.3024
lands on BM25's 0.2991, so halving the depth WITHOUT the teacher's signal gives
up the entire advantage over lexical retrieval.

### Fusion — and the third confirmation that errors are task-intrinsic
Reciprocal-rank fusion (k=60), weights tuned on dev:
  best single dense     0.3346
  **dense-only fusion   0.3435**  weights [2,1,1,0] (distil dropped)
  dense + BM25 fusion   0.3391

Adding BM25 makes fusion WORSE, refuting the expectation that the most
decorrelated member would contribute most. BM25 is maximally different from a
bi-encoder — different algorithm, different features, no shared parameters —
and still adds nothing beyond what the dense models already retrieve.

| phase | members | gain over best single |
|---|---|---|
| Ph3 RR | 4 fine-tunes of ONE checkpoint | +0.007 |
| Ph5 LSI | 5 models, 4 architectures | +0.0102 |
| Ph7 PCR | 4 architectures + a lexical ranker | +0.0089 |

Three phases, members ranging from near-identical to a different paradigm
entirely, and the gain is ~+0.01 every time. Phase 5 concluded the residual
errors are task-intrinsic rather than model-specific; Phase 7 is the strongest
test of that and it holds. What is left unretrieved is unretrievable by lexical
and dense methods alike — consistent with incomplete gold sets and with cases
whose relevance is not recoverable from the first 4,096 tokens.

Caveat on 0.3435: the weight grid searched ~324 configurations and took the
maximum over 118 dev queries. That is the Phase 3 selection bias again; the true
gain is smaller than +0.0089 and the leaderboard is the honest read.

### Submissions
| | model | dev F1@7 |
|---|---|---|
| p7_sub1 | full fine-tune, 4-chunk max-sim, in-batch negatives | 0.3346 |
| p7_sub2 | LoRMA 0.27% on prefix-LM | 0.3202 |
| p7_sub3 | top-4-layer on efficient arch | 0.3168 |
| p7_sub4 | distilled 6/12-layer student | 0.3157 |
| p7_sub5 | RRF fusion of 4 dense rankers | **0.3435** |
(BM25@4096 control, not submitted: 0.2991)

### Not done
Negatives from ranks 50-200 (the test of the incomplete-gold account);
listwise ranking objectives; chunk budgets other than 4,096 for the dense model;
cross-encoder reranking of the top-50.

### Phase 4 leaderboard results
| submission | dev | board | delta |
|---|---|---|---|
| full fine-tune, tail 1024 | 0.7124 | **0.6997** | -0.013 |
| LoRMA on prefix-LM (0.27%) | 0.6836 | 0.6967 | +0.013 |
| distilled student (61.5%) | 0.6911 | 0.6954 | +0.004 |
| head+tail 512/512 | 0.6875 | 0.6918 | +0.004 |
| encoder-decoder | 0.6342 | 0.6300 | -0.004 |
| head 1024 | 0.5429 | 0.5104 | -0.033 |

Mean |delta| 0.012, max 0.033; ordering preserved except within the top
cluster. Phase 3's dev (10 docs) could not rank models at all; Phase 4's
(1,988 balanced cases, sd ~0.011) ranks them reliably. The Phase 3 failure was
dev SIZE, confirmed rather than conjectured.

Top four span 0.0079 on the board — full fine-tuning (100% trainable), LoRMA
(0.27%), distillation (61.5%) and a different input window are statistically
tied. The adaptation technique does not matter here.

Position ablation holds on test: tail 0.6997 vs head 0.5104, gap **0.189**
(dev 0.170) — 24x the technique spread. Which 1024 tokens the model reads
dominates every other design choice tested.

### Phase 5 leaderboard results
| submission | dev | board | delta |
|---|---|---|---|
| 5-model ensemble | 0.4132 | **0.3978** | -0.0154 |
| chunked max-aggregation | 0.4114 | 0.3933 | -0.0181 |
| distilled student | 0.4030 | 0.3888 | -0.0142 |
| full fine-tune 4 ep | 0.4004 | 0.3849 | -0.0155 |
| top-4-layer, efficient | 0.4004 | 0.3833 | -0.0171 |
| LoRMA, prefix-LM | 0.3689 | 0.3557 | -0.0132 |
| encoder-decoder | 0.3512 | 0.3358 | -0.0154 |

PERFECT rank correlation across all seven, including the sub1/sub2 tie that dev
could not separate (0.4004 both) and the board splits by 0.0016.

Deltas are a CONSTANT -0.0156 (sd 0.0015), not noise. The per-label thresholds
were fitted on dev, so dev is optimistic by exactly what fitting 100 thresholds
buys in-sample: **threshold-fitting optimism = 0.0156**, measured rather than
assumed.

Method consequence: selection bias shifts the LEVEL without disturbing the
ORDER, provided it is constant across models. Phase 3's best-epoch selection on
10 documents inflated by ~0.03 AND varied per run, which is why its ranking
failed; Phase 5's fitting on 10,181 documents inflates more than Phase 3 in
absolute terms but harmlessly, because every model pays the same amount.

Both late-stage gains transferred at about half size: ensembling +0.0129 on the
board (dev +0.0102), chunked aggregation +0.0045 (dev +0.0084).
Micro 0.49 vs macro 0.39 on test confirms the long tail is still the loss.

### Phase 6 leaderboard results
| submission | dev | board | delta |
|---|---|---|---|
| distilled 6/12 student | 0.8273 | **0.8260** | -0.0013 |
| prefix-LM + LoRMA (its teacher) | 0.8290 | 0.8251 | -0.0039 |
| encdec full fine-tune | 0.8225 | 0.8183 | -0.0042 |
| decoder top-4-layer | 0.8084 | 0.8074 | -0.0010 |

Dev (50 generated summaries) tracked the board within 0.004 — BERTScore's
compressed range makes even a small sample a reliable ranker. Top two are tied
(0.0009 apart) and swap order.

The DISTILLED STUDENT scored highest: 6/12 layers, 61.5% of parameters, above
the teacher it was distilled from. Halving the depth costs nothing measurable.

Extractive baseline, stated carefully: lead-512 scored 0.8300 on DEV and cannot
be computed on test (no public references). Applying the models' own -0.003
dev->board offset projects ~0.827 on test vs the best model's 0.8260, so the
best abstractive model TIES trivial extraction rather than losing to it. Same
substantive conclusion: a 110M model on 6,528 judgments buys nothing over
printing the document's opening, because BERTScore over a 512-token window with
86% vocabulary overlap cannot tell them apart.

Total model spread 0.019 (0.8074-0.8260) against +0.040 from decoding
configuration alone on one checkpoint. The decoding flag mattered more than
every architecture and adaptation choice combined.

### Phase 7 leaderboard results
| submission | dev | board | delta |
|---|---|---|---|
| full fine-tune | 0.3346 | **0.2947** | -0.040 |
| distilled student (61.5%) | 0.3157 | 0.2835 | -0.032 |
| top-4-layer (25.7%) | 0.3168 | 0.2812 | -0.036 |
| LoRMA, prefix-LM (0.27%) | 0.3202 | 0.2764 | -0.044 |

Mean delta -0.038, much larger than Phases 4-6 (-0.012, -0.016, -0.003), and
the ordering partially scrambled (LoRMA 2nd -> 4th, distilled 4th -> 2nd).

Explanation is structural, not selection bias: the TEST POOL is 1,727
candidates against dev's 1,023. Precision@7 falls as the pool grows, so dev and
test do not pose equally hard retrieval problems. The other phases' dev and test
had matched difficulty, and their offsets were correspondingly small.

Board spread across all four techniques is 0.018 -- identical to dev's -- so the
reordering is noise over 237 queries. Technique again does not matter: full
fine-tuning, LoRMA (0.27%), top-4-layer (25.7%) and a 61.5% distilled student
are tied, against +0.061 from chunked max-sim representation and +0.173 from
the BM25 budget choice.