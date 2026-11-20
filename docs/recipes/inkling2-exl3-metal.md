# Inkling 2 Small with EXL3 experts on Metal

TensorFold serves `thinkingmachines/Inkling-2-Small` on one Mac with 256 GB of memory:
- the routed experts in EXL3 trellis form: a `kortexa-inkling2-exl3-v1` pack, mul1 codebook, per-expert widths of
  2 to 4 bits;
- everything else in the source checkpoint's bf16.

Text only, one stream; no draft head, prompt cache or image tower yet.

## Model directory

- `config.json`: the source checkpoint's, with
  `"quantization_config": {"quant_method": "exl3", "format": "kortexa-inkling2-exl3-v1"}`.
- `model-dense.safetensors`: every `model.llm.*` tensor of the source checkpoint except the routed experts
  (`*.mlp.experts.*`), bytes unchanged. That covers embeddings, norms, attention, short convolutions,
  attention-residual queries, routers and the unembedding.
- From the source: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `special_tokens_map.json`.
- `experts/`: the pack.
  - `checkpoint.json`: its format, `"status": "complete"`, and each layer's files and expert ranges.
  - `layers/NN/experts-FFF-LLL.safetensors`, with `experts.E.{w13,w2}.{trellis,suh,svh,mul1}`.

The engine refuses a pack that is incomplete, does not cover every expert once, or uses another codebook or width.

## Build

    zig build native -Doptimize=ReleaseFast                # the server: zig-out/native/bin/tensorfold-native
    zig build tf-inkling-run tf-inkling-kernels -Doptimize=ReleaseFast

`tf-inkling-kernels` checks every Inkling kernel against an FP64 reference at the same rounding points. It also
checks that each row's result is independent of the rows beside it, and that a NaN stays in its own row.

## Check and serve

    tf-inkling-run MODEL check                             # load; report expert widths and size
    tf-inkling-run MODEL exact ids.json 448                # steps, 16-row windows and one chunk: identical bits
    tensorfold-native serve MODEL --name inkling2 --context 32768 --parallel 1 --load-limit-gb 236

- The published 2.59 bpw pack loads as 190.2 GiB (14,976 K2 and 21,888 K3 expert projections) in about 31 s on an
  M5 Ultra.
- The load limit covers weights, scratch and caches. The default is 75% of RAM, which leaves no room for caches
  beside this pack on a 256 GB Mac.
  - `--load-limit-gb` sets the limit; `INKLING_LOAD_LIMIT_GB` is the fallback.
  - At 236 GB, about 27 GiB is left for caches.
- Caches take 192 KiB a token, because every layer keeps its whole context, plus 33 MiB of convolution rings a
  stream.

Measured on an M5 Ultra (256 GB):

| | |
|---|---|
| Prompt | about 66 tok/s (8,192 tokens in 125 s) |
| Decode, one stream | 9.0 tok/s after a 512-token prompt (110 ms of GPU time a token); 9.8 tok/s served (256-token replies, 2.3 s to the first token for a 116-token prompt) |

The experts run through the core EXL3 projection, which is what limits decode speed.

## Fidelity

Teacher-forced on a 62,526-token held-out set against the MXFP8 release as teacher:
- **KL:** 0.0325.
- **Comparison:** the same pack's experts in an MLX forward score KL 0.0341. TensorFold's is 0.0016 lower (95% CI
  −0.0035 … −0.0001).
- **Top-1 agreement:** 95.5%.
- **Per layer:** hidden states stay within 1.3× the difference between two correct bf16 implementations.

## Limits

- One stream; `--parallel` above 1 is refused.
- Draft requests are served as plain and copy-drafted rounds; there is no draft head yet.
- `--vision` is refused.
- The server does not yet split Inkling's reply channels: a thinking reply's `reasoning_content` starts with the
  raw channel markers (`<|message_model|><|content_thinking|>`). Send
  `chat_template_kwargs {"enable_thinking": false}`, or serve with `--no-thinking`, for replies without thinking.
