# GLM-5.3 architecture

This guide shows GLM-5.3's main text-generation architecture and the tiny random model used in the CPU preflight tests.

The diagrams describe the native `GlmMoeDsaForCausalLM` implementation in Transformers 5.18.0 and the inspected GLM-5.3 config. They do not imply that complete Heretic integration or full-checkpoint execution has been validated.

## 1. Whole model

```text
Input token IDs
      │
      ▼
Token embedding
154,880 tokens → vectors of size 6,144
      │
      ▼
┌───────────────────────────────────────────┐
│ 78 decoder layers                         │
│                                           │
│ Layers 0–2:  sparse attention + dense MLP  │
│ Layers 3–77: sparse attention + MoE MLP    │
│                                           │
│ Each layer keeps vectors of size 6,144     │
└───────────────────────────────────────────┘
      │
      ▼
Final RMSNorm
      │
      ▼
Language-model head
6,144 → 154,880 token scores
      │
      ▼
Select next token
```

**MoE** means *mixture of experts*. It selects a few expert networks for each token rather than running every expert.

**MLP** means *multilayer perceptron*: the feed-forward network inside a decoder layer. **RMSNorm** normalizes a vector using its root mean square and learned scaling weights.

Layer numbers in this guide start at zero.

## 2. Inside one decoder layer

```text
                         Input hidden state x
                                  │
                 ┌────────────────┴─────────────────┐
                 │                                  │
                 ▼                                  │
              RMSNorm                               │
                 │                                  │
                 ▼                                  │
       Sparse self-attention                        │
                 │                                  │
                 ▼                                  │
        o_proj  ◀── Heretic target                   │
                 │                                  │
                 └─────────────── + ◀───────────────┘
                                  │
                         Updated hidden state
                                  │
                 ┌────────────────┴─────────────────┐
                 │                                  │
                 ▼                                  │
              RMSNorm                               │
                 │                                  │
                 ▼                                  │
         Dense MLP or MoE MLP                       │
                 │                                  │
                 └─────────────── + ◀───────────────┘
                                  │
                                  ▼
                         Output hidden state
```

The bypass paths are **residual connections**. Each block adds its output to the running hidden state. That running vector is also called the *residual stream*.

Heretic targets the attention output projection, `self_attn.o_proj`, and the dense/shared/routed MLP output projections, `down_proj`. GLM-5.3's routed targets still require automatic Heretic integration; the preflight tests installed their wrappers manually.

## 3. Sparse attention

There are two separate ideas here: compressed key/value representations and sparse token selection.

```text
Normalized hidden states
          │
          ├── Query projections ───────────────► Q
          │
          ├── Compressed KV projection
          │              │
          │              ├── Compressed KV cache
          │              │
          │              └── Expand into ─────► K, V
          │
          └── Token-selection indexer
                         │
                         ▼
              Selected causal token positions
              Up to 2,048 positions per query
                         │
                         ▼
              Causal attention using Q, K, V
              restricted to selected positions
                         │
                         ▼
                  Concatenate heads
                         │
                         ▼
                       o_proj
                         │
                         ▼
              Vector of size 6,144
```

- **Q (queries)** represents what each token is looking for.
- **K (keys)** represents what each available token can be matched against.
- **V (values)** carries the information combined by attention.
- The **KV cache** retains earlier key/value information during token-by-token generation. This implementation caches compressed KV representations and expands them for attention.
- **DSA** means sparse attention with a token-selection indexer. The indexer selects positions; the attention block then computes attention over those positions.

Rotary position encoding is applied to part of the queries and keys. It tells attention where tokens occur. Causal masking prevents a token from attending to future tokens; the current token can be included.

The 78 layers contain:

- **21 full indexer layers:** compute a new token selection.
- **57 shared-indexer layers:** reuse the latest full layer's selection.

**Shared selection does not mean shared attention weights.** Those layers still compute their own queries, keys, values, and attention output.

The selected-position limit is not the model's context-window limit. The inspected config allows a much longer context; sparse selection limits how many positions each query attends to.

## 4. MoE block in layers 3–77

```text
                       Hidden state
                            │
           ┌────────────────┴─────────────────┐
           │                                  │
           ▼                                  ▼
     Shared expert                         Router
     Always runs                              │
           │                                  ▼
           │                      Select 8 of 256 experts
           │                      and their routing weights
           │                                  │
           │                 ┌────────────────┼───────────┐
           │                 ▼                ▼           ▼
           │              Expert A         Expert B      ...
           │                 │                │           │
           │                 └────────────────┼───────────┘
           │                                  │
           │                                  ▼
           │                         Weighted output sum
           │                                  │
           └──────────────── + ◀──────────────┘
                             │
                             ▼
                        MoE output
```

Each token gets its own expert selection. There is one shared expert per sparse layer, and it always runs. The routed experts run only for the tokens assigned to them.

Each expert uses this MLP calculation:

```text
Input x
   │
   ├── gate projection ── SiLU ──┐
   │                             × ── down_proj ── Output
   └── up projection ────────────┘
```

`×` is element-wise multiplication. **SiLU** is the activation function applied to the gate branch. Heretic targets `down_proj`, where the expert output returns to the hidden-state space. The dense MLP layers use the same basic gate/up/down structure, with a larger intermediate size.

For routed experts, the native implementation packs all weights into tensors:

```text
gate_up_proj: [256, 4,096, 6,144]
down_proj:   [256, 6,144, 2,048]
                  ▲
                  │
     Heretic's wrapper exposes each expert's
     down_proj as a hookable Linear module,
     sharing the original tensor storage.
```

The first dimension selects an expert. The other dimensions are output and input sizes. `gate_up_proj` combines the two projections, so its output dimension is twice the expert intermediate size: `2 × 2,048 = 4,096`.

The router itself is not an abliteration target in this plan.

## 5. The tiny model tested in preflight

```text
32-token vocabulary
        │
        ▼
Embedding: hidden size 16
        │
        ▼
Layer 0: FULL indexer   + DENSE MLP
        │
        ▼
Layer 1: SHARED indexer + MoE
         reuses layer 0's selection
        │
        ▼
Layer 2: FULL indexer   + MoE
        │
        ▼
Final RMSNorm
        │
        ▼
32 next-token scores
```

Each tiny MoE layer has **one shared expert and four routed experts**, with **two routed experts selected per token**.

| Setting | Real GLM-5.3 | Tiny test model |
| --- | ---: | ---: |
| Decoder layers | 78 | 3 |
| Hidden size | 6,144 | 16 |
| Dense MLP intermediate size | 12,288 | 24 |
| Expert intermediate size | 2,048 | 8 |
| Attention heads | 64 | 2 |
| Routed experts per sparse layer | 256 | 4 |
| Routed experts selected per token | 8 | 2 |
| Vocabulary size | 154,880 | 32 |
| Indexer selection limit | 2,048 | 2 |

Attention projection and indexer dimensions are also reduced. The full configuration is in `tests/glm_dsa_preflight.py`, in `tiny_model()`.

The tiny model uses the same native Transformers class, but starts with random weights:

```python
config = GlmMoeDsaConfig(...)  # Small, coherent dimensions and layer patterns.
model = AutoModelForCausalLM.from_config(config).eval()
```

`from_config()` initializes weights locally. It does not download the pretrained GLM-5.3 checkpoint. A fixed random seed makes the tests repeatable.

The tests compare the native tiny model against the same model with expert wrappers installed. They check math, hooks, shared storage, cached generation, and repack/save/reload behavior. The random model has no useful language ability; generated tokens only prove that generation machinery runs.

## 6. Heretic component coverage

From the resolved full-model config, the intended projection counts are:

| Component | Count |
| --- | ---: |
| Attention output projections | 78 |
| Dense MLP output projections | 3 |
| Shared expert output projections | 75 |
| Routed expert output projections | 19,200 |
| Total MLP output projections | 19,278 |

There are `75 × 256 = 19,200` routed projections. Only selected experts receive activation rows for a given prompt. Missing activation data for an expert must not be confused with a missing expert module.

## 7. Scope and limitations

These diagrams show the main 78-layer causal-language-model path. The published config also declares one additional next-token prediction layer (`num_nextn_predict_layers = 1`). That path is not covered by the current tiny-model tests.

The official BF16 and FP8 checkpoints use the same main architecture. Their weight formats differ. The planned unquantized target is `zai-org/GLM-5.3-BF16`.

Tiny-model success does not establish full-checkpoint memory use, GPU speed, multi-GPU correctness, optimization quality, refusal changes, or benchmark preservation. See `glm-5.3-preflight-report.md` for tested scope and `glm-5.3-plan.md` for the remaining implementation and 8×B300 validation gates.
