# On-device diffusion chat model

![status](https://img.shields.io/badge/status-in%20development-orange)
![python](https://img.shields.io/badge/python-3-3776AB?logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![kaggle](https://img.shields.io/badge/trained%20on-Kaggle%20T4-20BEFF?logo=kaggle&logoColor=white)
![params](https://img.shields.io/badge/params-~18M-6A4C98)

A small conversational language model built from scratch in PyTorch and designed to run fully on-device. Instead of the usual left-to-right decoder, the model is a **masked-diffusion encoder–decoder Transformer**: the dialogue context is encoded once, and the reply is produced by iteratively un-masking a fixed-size answer window.

The project is structured to walk through every stage of a production LLM lifecycle on a scale that fits a single free-tier GPU:

| Stage | What it means here | Status |
| --- | --- | --- |
| Data & tokenizer | corpus collection, cleaning, deduplication, byte-level BPE | done |
| Pretraining | masked-diffusion objective on 1.47B tokens of educational web text | done |
| Supervised fine-tuning | dialogue data, loss on the assistant turn only | done |
| Reinforcement learning | policy optimisation on top of the SFT model | coming soon |
| Benchmarking | standard zero-shot suites against reference models | coming soon |
| Production runtime | export and a C++ inference runtime | coming soon |

> This repository is a work in progress. It is published at the end of Sprint 2; sections marked *coming soon* describe planned work that is not in the code yet.

---

## Architecture

![Model architecture](docs/architecture.png)

The context and the answer live in two separate windows. The **encoder** reads the clean dialogue history once with bidirectional self-attention. The **decoder** sees the answer slot, in which a random fraction of tokens has been replaced with `[MASK]`, and reads the encoder output through cross-attention. Generation runs the decoder for several steps over the answer slot only, so the cost of a long context is paid once per message rather than once per step. The same split was later found to match the encoder–decoder design of E2D2 [2].

All blocks use the current standard Transformer recipe — pre-norm RMSNorm, SwiGLU feed-forward, rotary position embeddings — with attention assembled by hand from linear projections and `scaled_dot_product_attention` rather than taken from a library module. Input embeddings, the LM head and the vocabulary projection inside the Thinking module share one weight matrix.

| Property | Value |
| --- | --- |
| Objective | masked (absorbing-state) discrete diffusion, MDLM-style [1] |
| Width / heads | d_model 384, 6 heads × 64 |
| Depth | 4 encoder blocks, 2 decoder blocks |
| Context window | 256 tokens |
| Answer window | 128 tokens (pretraining), 64 tokens (SFT) |
| Vocabulary | 12 288, byte-level BPE, tied embeddings |
| Position encoding | RoPE in self-attention; none in cross-attention |
| Parameters | ~18M |

### Why diffusion for a phone

The target runtime is a mobile inference engine that works best with a static computation graph. A masked-diffusion model maps naturally onto one: the input is always `[1, L]`, the output is always `[1, L, V]`, and there is no KV-cache to manage between steps. The encoder–decoder split then keeps the per-step cost proportional to the 64-token answer window instead of the full 320-token sequence.

### Thinking module (experimental)

Between the encoder and the decoder sits a small module of my own design. The motivation was to give the decoder, alongside the usual token-level context, a coarse summary of *what kind of answer is expected*: each context position is compared with the whole vocabulary through cosine similarity against the shared embedding matrix, and the result is passed on through two channels — a per-position *words* channel and a pooled *thought* channel that summarises the context as a whole. In the released model the module replaces the raw encoder output.

---

## Data

| Split | Source | Size |
| --- | --- | --- |
| Pretraining | FineWeb-Edu, `sample-10BT` | 2.46B train / 12M validation tokens |
| SFT | DailyDialog, EmpatheticDialogues, PersonaChat, MBPP (sanitized), 1 308 synthetic chat and refusal dialogues | 43 732 / 692 / 704 dialogues |

Every source passes through its own adapter into a single schema, then through shared normalisation (`ftfy`, quote and dash unification), filtering, exact deduplication by hash and near-duplicate removal with MinHash LSH. Train / validation / test assignment is deterministic, derived from a hash of the record key, so the split is reproducible and leak-free across re-runs.

The tokenizer is a byte-level BPE with 12 288 entries. The size was chosen by comparing compression at 8k, 12k and 16k: on a phone the output projection is computed at every diffusion step, so vocabulary size has a direct latency cost. Five special tokens (`<pad>`, `<mask>`, `<user>`, `<bot>`, `<eos>`) are used, and sixteen more are reserved for later conditioning. The corpus is stored as `uint16` binary shards; SFT shards carry a parallel mask marking the assistant turns that are allowed to be noised.

## Training

Pretraining and SFT run the same step; only the data differs.

1. Sample a noise level `t ~ U(0, 1)` per example, using a low-discrepancy grid over the batch to reduce the variance of the estimate.
2. Replace each answer token with `[MASK]` independently with probability `t`.
3. Predict the original tokens; cross-entropy is computed on masked positions only and weighted by `1/t`, which gives the diffusion ELBO.

During pretraining a 384-token window from the corpus is split into 256 context tokens for the encoder and 128 continuation tokens for the decoder. During SFT the context is the dialogue history up to the `<bot>` tag and the target is the assistant reply. The loss covers the whole answer slot, padding included, so the model learns to place `<eos>` and choose the reply length itself.

| | Pretraining | SFT |
| --- | --- | --- |
| Initialisation | random | best pretraining checkpoint |
| Steps × batch | 40 000 × 96 | multi-epoch, checkpoint chosen by validation loss |
| Tokens seen | ~1.47B | ~5M per epoch |
| Answer window | 128 | 64 |
| Hardware | 1 × T4, ~8 h | 1 × T4 |

Common setup: AdamW with separate weight-decay groups, linear warm-up followed by cosine decay, fp16 autocast with gradient scaling and clipping, step-based schedule with periodic validation, and checkpoints that separate *initialise from* and *resume from*.

## Sampling

Replies are generated block by block inside the answer window. At each step the decoder predicts every masked position, the most confident predictions are committed, and the rest stay masked for the next step. On top of this I added a training-free remasking rule derived from the G-Star sampler [3]: committed tokens whose confidence falls below the current mean can be returned to `[MASK]`, limited by a per-step budget that decreases over the course of generation. A repetition penalty handles the looping failure mode typical of small diffusion models.

---

## Roadmap

### Sprint 1 — Data and tokenizer · done

Corpus collection, cleaning, deduplication, deterministic splits, BPE training, binary sharding.

### Sprint 2 — Pretraining, SFT, RL · in progress

- [x] Encoder–decoder masked-diffusion model
- [x] Pretraining
- [x] Supervised fine-tuning
- [x] Block-wise sampler with confidence-based remasking
- [ ] Reinforcement learning — *coming soon*

  Group-relative policy optimisation with rule-based rewards (termination, length, repetition, relevance) and a KL penalty to the SFT model. A diffusion model has no exact sequence log-likelihood, so the policy ratio is estimated from the masked-diffusion loss with noise masks shared between the policy and the reference model, following d1 [4] and VRPO [5].

### Sprint 3 — Benchmarking · coming soon

- Zero-shot evaluation with `lm-evaluation-harness` on HellaSwag, PIQA, ARC-Easy / Challenge, WinoGrande and LAMBADA, using a diffusion-specific log-likelihood estimator.
- Comparison with GPT-2 small and with the parameter-matched baseline without the Thinking module.
- Effect of each lifecycle stage: pretrained vs. SFT vs. RL checkpoints.

### Sprint 4 — Production runtime · coming soon

- Export PyTorch → ONNX → ncnn with numerical parity checks at every stage, as in the earlier [on-device detector](https://github.com/quark28/person-mobile-detector).
- C++ inference runtime: tokenizer, sampler and diffusion loop around the inference engine.
- Chat application on top of the runtime.

---

## Repository layout

```
sprint1/                        data and tokenizer
  data-preprocessing.ipynb      raw source collection, per-source adapters to a unified schema
  data-processing.ipynb         normalisation, filtering, deduplication, deterministic splits
  tokenizer.ipynb               BPE training, round-trip checks, binary sharding
sprint2/                        model and training
  pretrain.ipynb                model definition, masked-diffusion pretraining
  sft.ipynb                     supervised fine-tuning, test-set evaluation, sampling
docs/                           architecture diagram and its source
```

All notebooks were run on Kaggle. Model weights and the tokenised corpus are not stored in the repository.

## References

1. Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
2. Arriola et al. *Encoder–Decoder Diffusion Language Models for Efficient Training and Inference.* NeurIPS 2025.
3. G-Star remasking sampler. arXiv:2510.08369.
4. Zhao et al. *d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning.* 2025.
5. Zhu et al. *LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models.* 2025.
6. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
7. Shazeer. *GLU Variants Improve Transformer.* 2020.
