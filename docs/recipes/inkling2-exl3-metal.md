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
- `INKLING_ROUND_TIMING=1` logs, when each reply ends, what an average round spent: GPU and encode time, the rest of
  the forward call (commit and wake), the draw, and the time between rounds. Use it to find where a slow reply
  loses its time; it adds a few clock reads to each round.
- A decode step's host side has two switches, both on by default and both bit-neutral. `INKLING_WARM_WAIT=0` makes
  the step wait for its command buffer by blocking, as before: by default it sleeps through all but the last 15 ms of
  the GPU time that steps of the same row count recently took and spins on the command buffer from there, because a core
  that blocked for the whole step runs the draw after it several times slower. `INKLING_FIRST_CHUNK=N` sets how many
  layers the step commits in a command buffer of their own before the rest is encoded (default 4; 0 is one command
  buffer), so the GPU starts while the host is still encoding.
- `tf-inkling-run MODEL wall LAYERS POS ROWS STEPS [REPEAT]` runs decode steps each followed by the greedy draw and
  prints the median wall time and its parts (encode, commit to GPU start, GPU, GPU end to wake, draw), with a digest
  of the last hidden rows to compare two configurations bit for bit. `tf-inkling-kernels time` times the fused norm
  kernels at the block and query counts of the real layers.

Measured on an M5 Ultra (256 GB):

| | |
|---|---|
| Prompt | about 65 tok/s (8,202 tokens in 126 s to the first token) |
| Decode, one stream, served | 38.8 tok/s with a 116-token prompt (256-token replies, 2.2 s to the first token); 37.9 tok/s after an 8,202-token prompt |
| Decode, sampled (temperature 0.7, top_p 0.95) | about 32 tok/s |

Decode steps of up to 16 rows take a decode path: the experts run a rate-grouped EXL3 chain (adapted from Sushi), and
attention, norms and the attention-input matvecs have their own kernels. A step takes about 25 ms of GPU time. Prompt
rows still take the per-operation kernels, which is what limits prompt speed.

## Fidelity

Teacher-forced on a 62,526-token held-out set against the MXFP8 release as teacher:
- **KL:** 0.0325 on the prompt path; 0.0336 with every row after the first 16 fed through the decode path in 16-row
  windows.
- **Comparison:** the same pack's experts in an MLX forward score KL 0.0341. TensorFold's is 0.0016 lower on the
  prompt path (95% CI −0.0035 … −0.0001) and 0.0005 lower on the decode path (95% CI −0.0020 … +0.0011).
- **Top-1 agreement:** 95.5%.
- **Per layer:** hidden states stay within 1.3× the difference between two correct bf16 implementations (1.21× on the
  decode path).
- **Decode windows:** a window of up to 16 rows gives the same bits as the same rows decoded one at a time.

## Limits

- One stream; `--parallel` above 1 is refused.
- Draft requests are served as plain and copy-drafted rounds; there is no draft head yet.
- `--vision` is refused.
- The server does not yet split Inkling's reply channels: a thinking reply's `reasoning_content` starts with the
  raw channel markers (`<|message_model|><|content_thinking|>`). Send
  `chat_template_kwargs {"enable_thinking": false}`, or serve with `--no-thinking`, for replies without thinking.
