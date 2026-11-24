# Inkling 2 Mini (`inkling2_loop`, native NVFP4/MXFP8) on Metal

TensorFold serves `thinkingmachines/Inkling-2-Mini` directly from the Hugging Face checkpoint on one Mac with 32 GB or more of unified memory (128 GB fits full 131k contexts and multi-stream serving with wide cache headroom):
- the routed experts in the checkpoint's native ModelOpt weight-only formats — NVFP4 (block-16 E4M3 scales + per-expert F32 global scale) on physical layers 0–19 and MXFP8 (block-32 UE8M0 scales) on physical layers 20–23 — with no repacking or re-quantization;
- dense projections, short convolutions, norms, attention-residual queries, rank-64 cross-pass `MessageGate` controllers, routers, and unembedding in bf16;
- the 24 physical layers unrolled into 48 virtual layers (`6` prelude + `12 × 3` looped core passes + `6` postlude), each virtual layer owning its own KV and short-convolution state.

Text only, one to eight concurrent streams (`--parallel 1` through `8`, default `1`); drafts come from the checkpoint's 3-layer DSpark draft head (kept in bf16 by default), with an in-memory prompt cache for multi-turn reuse and chunked prefill interleaving (`--decode-share`). The checkpoint's `model.visual.*` tensors are skipped at load (`--vision` is refused).

## Model directory

Point TensorFold directly at the downloaded `thinkingmachines/Inkling-2-Mini` snapshot directory:

- `config.json`: `"model_type": "inkling2_loop"`, `"quantization_config": {"quant_method": "modelopt", ...}`, `loop_config` (`num_prelude_layers: 6`, `num_core_layers: 12`, `num_core_passes: 3`, `num_postlude_layers: 6`), and `mtp_config` (3 layers, 4 taps `[5, 23, 41, 48]`).
- `model.safetensors.index.json` + sharded `model-*-of-*.safetensors`: `model.llm.*` tensors are loaded directly; `model.visual.*` tensors are ignored.
- `mtp.safetensors`: the checkpoint's 3-layer DSpark draft head, bytes unchanged. Serving with drafts loads it; `--no-drafts` serves without it.
- Tokenizer and chat template files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `special_tokens_map.json`).

## Build

    zig build native -Doptimize=ReleaseFast                # the server: zig-out/native/bin/tensorfold-native
    zig build tf-inkling-run tf-inkling-kernels tf-inkling-bench -Doptimize=ReleaseFast

`tf-inkling-kernels` checks every Inkling kernel — including GQA=3 attention (`attn_split`, `pattn`), NVFP4/MXFP8 decode and tensor-unit routed experts (`fp_experts`, `xp_fp`), 1-slot and fused 3-slot `MessageGate` controllers (`controllers`), and 4-tap GQA=3 DSpark kernels (`dspark`) — against FP64 references at the exact rounding points, along with row independence and NaN containment.

## Check and serve

    tf-inkling-run MODEL check                             # load; report physical/virtual layers, precision and size
    tf-inkling-run MODEL exact ids.json 256                # 1-row steps vs 16-row windows: identical bits
    tf-inkling-run MODEL partition ids.json 17 200 --restore 17 --restore 217 --drafts
    tf-inkling-run MODEL shared ids.json 1                 # 2/4/8-stream shared forward vs solo + order invariance
    tensorfold-native serve MODEL --name inkling2-mini --parallel 4

- The checkpoint loads in about 3.8 s on an M5 Max: `17.59 GiB` for the main model (`18.3 GiB` total with the bf16 DSpark draft head for 8 streams), `1.45 GiB` of prompt-chunk scratch at the default 4,096 rows, and `19.8 GiB` wired at rest.
- On a 128 GB Mac (default 75% load limit), `74.1 GiB` remains for KV caches and prompt snapshots — `1,048,576` token slots across up to 8 streams, with `8.0 GiB` reserved for the prompt cache (`--prompt-cache-gib`, `0` disables it). On a 32–64 GB Mac, lower `--load-limit-gb` or `INKLING_PREFILL_ROWS=512` if tighter scratch/cache bounds are needed.
- Virtual layer geometry mirrors Small's KV footprint per token: 40 sliding virtual layers (1,024-token window, ring sized to `alignForward(window + prefill_rows + 64, 256)`) and 8 global virtual layers (`8` KV heads × `128` dims in bf16 = `32 KiB/token`). Each admitted stream reserves the fixed sliding-ring and convolution floor plus `32 KiB/token` dynamically.
- **Prompt chunk path (M5 tensor units):** on GPUs with tensor units, prompt chunks take `.chunk` (`INKLING_PREFILL_ROWS`, default `4,096`): dense projections, routers, and Pass 1–2 `MessageGate` `forget`/`gain` projections (`ctrl_nax`) run as tensor-unit GEMMs, attention runs parameterized `pattn` (`NQ=24, G=3`), and routed experts run `xp_fp` (single-list GPU sort into 32-row windows with exact in-register bf16 dequantization of both NVFP4 `e2m1 × e4m3` and MXFP8 `e4m3 × ue8m0` into `mma_16x32`, `SoftcappedSwiGLU`, and pick-ordered combine). GPUs without tensor units (or `INKLING_PROMPT_PATH=per_op`) fall back to the per-operation prompt path (`512`-row default).
- **Decode path:** single-stream decode steps (`1..16` rows, including draft verification) use per-row NVFP4/MXFP8 expert kernels at 1 row and bit-identical GPU-grouped (`<= 4`-slot tile) kernels (`ik_fp_group` + `*_g`) at `2..16` rows, GQA=3 split-K attention (`attn_split` with `ik_attn_part_w_g3` shared K/V loads at long contexts), and fused Pass 1–2 decode controllers (`ik_loop_attn_prep`/`prep3` & `ik_loop_mlp_prep` → stacked `mvs` `down_proj` → coalesced multi-threadgroup `ik_msggate_fg_decode`/`fg_decode3` + `ik_norm3_f32`), keeping 1-row steps, 16-row windows, and drafted verification bit-identical. Wide multi-stream shared decode batches (`segments.len > 1` and `rows >= 12`, up to 64 rows) switch routed experts to `xp_fp`, and concurrent DSpark holds batch their draft projections, unembedding, and `fold_w` Markov scoring across active streams (`ikd_fold_pick_batch`, with lazy `27.9 MiB` batch scratch allocated only when `c > 1`).
- **Host-side switches:** `INKLING_WARM_WAIT=0`, `INKLING_FIRST_CHUNK=N`, `INKLING_ROUND_TIMING=1`, `INKLING_PREFILL_ROWS`, `INKLING_PROMPT_PATH`, `--decode-share`, and `launchd` settings (`ProcessType: Interactive`) behave identically to [`inkling2-exl3-metal.md`](inkling2-exl3-metal.md).

## Drafts

The checkpoint's 3-layer DSpark draft head (`H=2304`, `I=9216`, `Q=24`, `KV=8`, 4 taps `[5, 23, 41, 48]`) proposes up to 7 tokens a round, verified in one window so drafted replies match plain decoding token for token.

- **Default precision:** when a draft head's layer projections fit in `1 GiB` (Mini's 3-layer head is `~0.52 GiB` of projections, `0.72–0.80 GiB` total with norms, Markov head, and lane rings), TensorFold keeps the draft layers in **bf16 (`16`-bit)** by default instead of quantizing to 8-bit. Setting `INKLING_DRAFT_BITS=8` switches the draft layers to MLX affine 8-bit (saving `0.23 GiB` and `~0.33 ms` on a depth-7 hold — `4.27 ms` vs `4.60 ms` on M5 Max — because the bf16 Markov projection `201,024 × 256` dominates hold bandwidth).
- `INKLING_DRAFT_CONFIDENCE` (default `0.7`, or `off`) and `INKLING_DRAFT_DEPTH` (`1..7` or `auto`) work identically to Small.

## Measured performance

Measured on an Apple M5 Max (40-core Metal 4 GPU, 128 GB unified memory, ~614 GB/s) at default settings (`.chunk` 4,096 rows, bf16 DSpark on unless noted):

### Direct prefill and decode (`tf-inkling-run`)

| Workload | Plain (`--no-drafts`) | Drafted (bf16 default) | Notes |
|---|---:|---:|---|
| Prefill, `512` tokens (warm) | `2,175 tok/s` (`235 ms`) | — | `norm: 40.4 ms`, `experts: 117.6 ms` |
| Prefill, `8,192` tokens (warm) | `2,754 tok/s` (`2.97 s`) | `2,762 tok/s` (`2.97 s`) | `absorb: 8.8 ms`; bit-identical hidden digest |
| Prefill, `32,704` tokens | `2,282 tok/s` (`14.33 s`) | — | 8 chunks |
| Prefill, `131,008` tokens | `1,601 tok/s` (`81.8 s`) | — | 32 chunks (sustained laptop thermal envelope) |
| Decode, `512` + `128` tokens | `61.5 tok/s` (`16.0 ms` GPU) | `76.0 tok/s` (`1.23×`) | `128/128` bit-identical (`2.06` tok/round on this passage) |
| Decode, `32,704` + `64` tokens | `51.4 tok/s` (`19.2 ms` GPU) | `64.2 tok/s` (`1.25×`) | `64/64` bit-identical; `71.1%` acc, `2.00` tok/round |
| Decode, `131,008` + `64` tokens | `37.1 tok/s` (`26.6 ms` GPU) | `49.0 tok/s` (`1.32×`) | `64/64` bit-identical; `63.8%` acc, `1.94` tok/round |

### Served concurrency and prompt cache (`tensorfold-native serve --parallel 8`)

Loaded in `3.8–4.7 s` (`18.3 GiB` weights incl. `0.80 GiB` bf16 DSpark, `1.45 GiB` scratch, `74.1 GiB` caches, `19.8 GiB` wired, `18.5 GiB` peak RSS). Distinct ~245-token prompts with `max_tokens: 128` (`temperature: 0`) for the concurrency sweep; `32,046`-token prompt with `max_tokens: 64` for the 32k cache check:

| Workload | Concurrency | Prompt tokens | TTFT | Decode (`tok/s`) | Aggregate decode (`tok/s`) | Notes |
|---|---:|---:|---:|---:|---:|---|
| Short decode (`c=1`) | 1 | 221 | `0.33 s` | `87.5` | `87.5` (`1.00×`) | — |
| Short decode (`c=2`) | 2 | 221–248 | `0.35 s` | `57.7` | `115.3` (`1.32×`) | — |
| Short decode (`c=4`) | 4 | 221–269 | `0.57 s` | `33.2` | `132.7` (`1.52×`) | `113.1 tok/s` agg wall |
| Short decode (`c=8`) | 8 | 221–269 | `1.16 s` | `21.8` | `174.1` (`1.99×`) | `121.2 tok/s` agg wall |
| 32k cold (`c=1`) | 1 | 32,046 | `12.66 s` | `95.9` | `95.9` | `3.56` tok/round (`71.4%` acc) |
| 32k cached, exact hit (`c=1`) | 1 | 32,046 (`32,045` hit) | `0.16 s` | `96.4` | `96.4` | `80.8×` lower TTFT; identical output SHA |
| 32k cached, new suffix (`c=1`) | 1 | 32,043 (`28,705` hit) | `1.61 s` | `84.1` | `84.1` | `7.9×` lower TTFT |

Shared forward calibration at startup (48 virtual layers, tapped): single-stream `1..8` rows `16.6..32.7 ms`; multi-stream shared `16` rows `45.8 ms`, `24` rows `55.4 ms`, `32` rows `64.5 ms`, `48` rows `84.3 ms`, `64` rows `102.8 ms`.

## Fidelity

Checked against the `mlx-lm` reference implementation of `inkling2_loop` over 512 tokens across all 48 virtual layers:

- **Per-layer hidden states:** worst ratio to the bf16 reference noise floor is `1.31×` on the tensor-unit chunk path (`1.34×` on the decode window path; gate `1.5×`).
- **Held-out KL & top-1:** KL `0.00114` (95% CI upper `0.00135`, gate `<= 0.002`) and `98.63%` top-1 agreement (`>= 98%`) in chunk mode; KL `0.00095` (95% CI upper `0.00106`) and `98.83%` top-1 in decode window mode.
- **Invariance:** `exact 256` reports `0` differing elements across all 48 virtual layers; `partition 17 200 --restore 17 --restore 217 --drafts` reports `0` differences across KV caches, convolution rings, expert picks, taps, finals, logits, and all 7 first-hold draft proposals; `shared 1` reports `order_diff: 0`, `solo_diff: 0` (2-stream), and `nonfinite: 0`.

## Limits

- Up to 8 concurrent streams (`--parallel 1` through `8`, default `1`); values outside `1..8` are refused.
- `--vision` is refused (`model.visual.*` tensors are skipped at load).
- Like Small, the server does not yet split Inkling's raw channel markers (`<|message_model|><|content_thinking|>`). Send `chat_template_kwargs {"enable_thinking": false}`, or serve with `--no-thinking`, for replies without thinking.
