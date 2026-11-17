# RUTAM v1 LFM3 Bridge Spec

Scope: research specification for V1-009. This records the published `LiquidAI/LFM3-1.7B-Base` implementation and a single proposed global-attention fusion point. It does not claim the real checkpoint has been loaded or run.

## Source revision and evidence labels

The Hugging Face model API reported repository revision `7bca5038823c0d0f14efe06439109354a10e014c` on 2026-11-17. Links below pin that revision. **Published** means directly stated in the model card, config, or model source. **Inference** means a bridge recommendation derived from those published facts. **Local repo** means observed in this checkout.

- [Model card and architecture summary](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/tree/7bca5038823c0d0f14efe06439109354a10e014c)
- [Serialized `config.json`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/config.json)
- [Model implementation `modeling_lfm3.py`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/modeling_lfm3.py)
- [Configuration implementation `configuration_lfm3.py`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/configuration_lfm3.py)
- [Convolution implementation `lfm3_conv.py`](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/lfm3_conv.py)
- [Hugging Face model metadata API](https://huggingface.co/api/models/LiquidAI/LFM3-1.7B-Base)

## Published model facts

| Fact | Published value |
|---|---|
| Loader mapping and top-level class | `architectures: ["Lfm3ForCausalLM"]`; `AutoModelForCausalLM` maps to `modeling_lfm3.Lfm3ForCausalLM`. |
| Module path | `Lfm3ForCausalLM.model` is an `Lfm3Model`; its decoder stack is `model.layers`, a `ModuleList` of `Lfm3DecoderLayer`. The attention submodule on attention layers is `self_attn: Lfm3Attention`. |
| Size | `hidden_size=2048`; `num_hidden_layers=24`; `intermediate_size=7680`; `vocab_size=128000`; `head_dim=128`. |
| Attention widths | `num_attention_heads=16`; `num_key_value_heads=4`, so this is grouped-query attention. |
| Layer types, zero-based | `conv` at 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23; `sliding_attention` at 4, 8, 12, 16, 20; `full_attention` at 2, 6, 10, 14, 18, 22. The model card summarizes these as 13 gated short-convolution, 5 sliding-window attention, and 6 global-attention layers. |
| Window and position limit | `sliding_window=256`; `max_position_embeddings=262144` (256K). |
| RoPE | `full_attention`: default RoPE, `rope_theta=1,000,000`, `partial_rotary_factor=0.5`; `sliding_attention`: default RoPE, `rope_theta=10,000`; convolution layers have no RoPE entry. `Lfm3RotaryEmbedding` implements the partial rotation as a zeroed tail, preserving a 128-wide head. |
| Declared dtype and tied embeddings | `dtype="bfloat16"`; `tie_word_embeddings=true`. `Lfm3ForCausalLM._tied_weights_keys` ties `lm_head.weight` to `model.embed_tokens.weight`. |
| Exact parameter count | Hugging Face metadata reports 1,728,801,536 parameters, all stored as BF16; the model card rounds this to 1.73B. |

The config and implementation are the authority for values above. The model card also reports text-only base-model status and 256K context. The `transformers_version` field in the config is `5.12.1`.

## Decoder-layer contract

**Published:** `Lfm3DecoderLayer.forward` has this argument order and returns a single tensor:

```python
forward(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    seq_idx: torch.IntTensor | None = None,
    **kwargs,
) -> torch.Tensor
```

Expected hidden input and output shape is `[batch, sequence, 2048]`. The decoder layer does **not** return `(hidden_states, attn_weights, present_kv)`. `Lfm3Attention.forward` returns `(attn_output, attn_weights)`, but the decoder layer unpacks it as `hidden_states, _` and discards the weights. The cache is a mutable Transformers `Cache` object passed as `past_key_values`; attention calls `past_key_values.update(key_states, value_states, layer_idx)` inside the attention forward. No `present_kv` is returned from the decoder layer.

Published residual and normalization order within `Lfm3DecoderLayer.forward`:

1. Save `residual = hidden_states`.
2. Apply `operator_norm` before the operator. On attention layers, run `self_attn`; on convolution layers, run `conv`.
3. With `use_sandwich_norm=true`, apply `post_operator_norm` to the operator output, then add the saved residual.
4. Apply `ffn_norm`, run `feed_forward`, and with sandwich norm apply `post_ffn_norm` to the MLP output.
5. Add the MLP output to the residual stream and return the resulting hidden-state tensor.

The attention module itself applies Q/K RMSNorm, RoPE, mutable cache update, causal or sliding attention, a sigmoid per-head output gate, and `o_proj`. The MLP is gated SiLU: `w2(silu(w1(x)) * w3(x))`. These details are why fusion should replace the completed decoder-block output, not the inner attention result.

## Single global-attention fusion point

**Inference / bridge choice:** use zero-based layer index **2**, the first `full_attention` block. Its expected runtime path from the public causal-LM model is:

```text
hf_model.model.layers[2]  # Lfm3DecoderLayer, config.layer_types[2] == "full_attention"
```

Attach a forward hook to that one block and replace the block's returned hidden tensor `[B, T, 2048]` with the fused tensor of the same shape and dtype. A hook preserves the module path and state-dict keys. This is after the attention/convolution operator, its optional sandwich norm, the first residual addition, the MLP, its optional sandwich norm, and the second residual addition. Do not replace `Lfm3Attention`'s tuple or touch its `past_key_values` argument.

Layer 2 is selected because the serialized config identifies it as the earliest global-attention layer. This allows graph fusion after the first full-context attention block while avoiding a layer-type guess based only on the generic `layers` list. This is a design choice, not a runtime-verified quality claim.

### Cache behavior

**Inference from published source:** the layer-2 cache update occurs inside `self_attn` before `Lfm3DecoderLayer.forward` returns. A post-layer replacement leaves the mutable cache object and the attention-produced K/V tensors untouched. Keep the hook active on every invocation, including one-token cached decode, and pass every original positional and keyword argument through unchanged. The modified hidden states then feed layers 3 onward; their caches are computed from those fused activations as part of normal execution. Do not separately fuse the prefill output and replay it into the cache.

When no graph context is supplied, return the original hidden tensor unchanged. When fusion is active and the graph module is FP32 while the backbone is BF16, perform the fusion in FP32 and cast only its result back to the original hidden dtype. Preserve the input device, sequence length, and hidden width. This dtype bridge is an implementation recommendation, not behavior verified against the model runtime.

### Attribute fallback and fail-closed behavior

**Inference / recommendation:** for the public `Lfm3ForCausalLM`, first resolve `hf_model.model.layers`. If the caller explicitly supplies a bare `Lfm3Model`, allow its direct `.layers`. Then require all of the following before attaching: 24 layers, `config.layer_types[2] == "full_attention"`, layer 2 is `Lfm3DecoderLayer`, and layer 2 exposes `self_attn` of type `Lfm3Attention`. If the documented path is missing or any check fails, the bridge constructor raises `ValidationError` before installing a hook; the caller can keep using the original model as baseline. Do not silently pick a different layer or substitute `SyntheticBackbone`.

## Hybrid-block quirks that matter to a bridge

**Published:** the stack is heterogeneous. `Lfm3DecoderLayer` owns either `self_attn: Lfm3Attention` or `conv: Lfm3ShortConv`; every layer also has its own feed-forward block and residual/norm path. The 13 convolution blocks are gated short causal convolutions with a width-7 state window (`conv_L_cache=7`, `conv_bias=false` for this checkpoint). Their forward path updates convolution state through the same Cache object. Five attention blocks use a 256-token sliding window; only six are full/global attention.

**Inference:** a wrapper that merely sees `backbone.layers` can preserve a common tensor-return API, but it cannot establish that an index is a global-attention block. Verify `config.layer_types[index]` and the actual layer/attention classes. The existing integration-guide example `fusion_locations: [1, 3]` points to convolution layers for this serialized layer pattern; it does not select global attention. The model uses short convolution, not a Mamba block in the inspected published source.

## Local integration gaps found in this checkout

**Local repo:** the default `RUTAMLFM3Adapter` still constructs `SyntheticBackbone` when no backbone is passed. Generic `RUTAMModelFactory.create_model` requires the supplied backbone to satisfy `BackboneProtocol`, while the public HF `Lfm3ForCausalLM` exposes the stack under `.model.layers` rather than the protocol getters or a top-level `.layers`. `inspect_backbone` recognizes a direct `.layers` plus `.config`, not this nested path. A separate explicit `RUTAMModelFactory.create_lfm3_bridge` entry point has now been added; it leaves generic `create_model` unchanged.

There is a second incompatibility beyond discovery: `execute_rutam_forward` calls `get_input_embeddings()` and then manually invokes every decoder layer, passing at most `attention_mask`. LFM3 prepares position IDs, rotary embeddings, per-layer masks, and the mutable cache in `Lfm3Model.forward`; its attention block unpacks `position_embeddings` and cannot run correctly when called with the layer's default `None`. **Inference:** implementing only the protocol getters would still bypass required LFM3 behavior. `RUTAMLFM3Bridge` now keeps the supplied model's outer HF `forward`/`generate` path in control and installs fusion at the nested layer-2 hook. Focused test doubles exercise that contract, but do not establish compatibility with the real Transformers implementation.

The current adapter also calls `validate_tensor(..., torch.float32)` on the backbone hidden states, and that helper requires a dense CPU FP32 tensor. This conflicts with the published BF16 checkpoint dtype. Preserve a no-graph identity path; on an active graph path, make the fusion dtype conversion explicit and restore the backbone dtype before returning to later layers.

### Implemented bridge surface

**Local repo:** `src/rutam/model/lfm3_bridge.py` resolves the documented `Lfm3ForCausalLM.model.layers[2]` structure, requires the 24-layer config and layer-2 full-attention type, and fails closed on unsupported layouts. It installs a post-layer hook, forwards model arguments unchanged, performs graph query/selection/encoding/traversal/fusion in CPU FP32, and casts the replacement activation back to the backbone output dtype. Package exports and `RUTAMModelFactory.create_lfm3_bridge` expose this as an explicit construction path. The bridge filters node visibility by authorization scope, read permission, verification state and validity window, then pins one graph snapshot across a `generate` call.

**Test-double evidence:** `tests/test_lfm3_bridge.py` covers resolver rejection, BF16 activation shape/dtype preservation, layer argument and mutable cache identity, graph-dependent fusion with a forced-open test gate, prefill plus cached decode while the live graph changes, access filtering, and explicit rejection of unsupported factory/generation modes. The tests use local doubles with Transformers-like class names and do not import Transformers. They demonstrate wiring only, not a real model load or learned retrieval quality.

## Local generation feasibility

**Published inputs:** metadata gives 1,728,801,536 BF16 parameters and the model file is listed at about 3.46 GB. **Inference from parameter count:** weights alone require 3,457,603,072 bytes (3.22 GiB) in BF16 or 6,915,206,144 bytes (6.44 GiB) in FP32, excluding Python/PyTorch runtime, loading overhead, temporary activations, and KV/convolution state.

For a batch-1, 128-token prompt, attention KV alone is about 5.5 MiB at FP32 or 2.75 MiB at BF16: 11 attention layers × K/V × 4 KV heads × 128 head dimensions × 128 tokens × element size. The short-prompt cache is small relative to the weights; long context is a different memory regime because six full-attention layers retain global K/V history.

At inspection time, the host reported 11 GiB RAM, 3.7 GiB available, and 3.0 GiB swap already in use. The i3-8130U reports AVX/AVX2 and no AVX-512/BF16 CPU flag. **Feasibility assessment:** BF16 has the best chance of fitting after freeing memory, but its 3.22 GiB weights leave little margin against the currently available RAM and native BF16 performance is not established on this CPU. FP32 weights alone exceed the currently available RAM by about 2.7 GiB; it is not a reliable resident-load target without clearing memory and using a low-peak-memory loader, and swapping will make generation painful. Short-prompt generation is plausible in principle, especially BF16, but was not run and cannot be claimed as locally validated.

## What remains unverified

- No model weights were downloaded or modified.
- `transformers` is not installed in the project virtual environment, and no real local LFM3 checkpoint is configured. No `from_pretrained`, generation, cached-decode, logits-parity, or real-model integration check was run.
- The contract test runs one prefill and two cached single-token decode calls on a local double. Real checkpoint generation and real cache/no-cache parity are unverified. Multi-beam or multiple-return generation is explicitly rejected until beam/cache reordering is integrated.
- Prefill currently loops over tokens and recomputes the query head on each growing prefix, so query work grows quadratically with prompt length. No latency or long-context memory benchmark was run.
- The guarded `test_real_lfm3_local_integration` now targets `create_lfm3_bridge` and, when a local checkpoint is configured, checks empty-graph logits parity, graph effect, and cached generation. It was skipped here because no checkpoint is configured. The new bridge has not been run against a real Transformers LFM3 model.
- CPU BF16 operator support and actual token throughput remain unmeasured. The RAM figure above is a point-in-time check and can change with other processes.

## Suggested bridge acceptance gate

Treat this document as an implementation contract, not proof. Validate the new resolver and fusion point against a user-provided local checkpoint only: baseline logits parity with fusion disabled, layer-2 output shape/dtype preservation, graph effect at layer 2, a prefill plus at least two cached one-token decode steps, and cache/no-cache output consistency. Record actual Transformers version and CPU dtype support in that run.
