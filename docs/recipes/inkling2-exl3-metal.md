# Inkling 2 Small with EXL3 experts on Metal

TensorFold serves `thinkingmachines/Inkling-2-Small` on one Mac with 256 GB of memory:
- the routed experts in EXL3 trellis form: a `kortexa-inkling2-exl3-v1` pack, mul1 codebook, per-expert widths of
  2 to 4 bits;
- everything else in the source checkpoint's bf16.

Text only, one stream; drafts come from the checkpoint's DSpark draft head, with an in-memory prompt cache for multi-turn reuse. No image tower yet.

## Model directory

- `config.json`: the source checkpoint's, with
  `"quantization_config": {"quant_method": "exl3", "format": "kortexa-inkling2-exl3-v1"}`.
- `model-dense.safetensors`: every `model.llm.*` tensor of the source checkpoint except the routed experts
  (`*.mlp.experts.*`), bytes unchanged. That covers embeddings, norms, attention, short convolutions,
  attention-residual queries, routers and the unembedding.
- From the source: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `special_tokens_map.json`.
- `mtp.safetensors`: the source checkpoint's DSpark draft head, bytes unchanged. Serving with drafts loads it;
  `--no-drafts` serves without it.
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
  - At 236 GB, 22.9 GiB is left for caches with the draft head (192.1 GiB of weights and 2.67 GiB of prompt-chunk
    scratch at the default 4,096 rows): enough for about 693k tokens of context. A 512-row chunk takes 0.37 GiB of
    scratch instead.
- Sliding layers (40 of 48, 1,024-token window) keep a ring of keys and values sized to `alignForward(window + prefill_rows + 64, 256)` slots (5,376 slots at the default 4,096-row chunk, 1,792 at 512), while the 8 global layers keep every position. Past the ring, context grows at 32 KiB a token (plus ~1.5 KiB of split-attention partials), over 0.87 GiB of sliding rings and convolution state at the default chunk (a 131,072-token stream takes 5.05 GiB).
- When `--context` is unset, the server advertises up to 131,072 tokens (or less if fewer fit); `--context` accepts any length up to what the caches fit, which the startup line reports.
- Prompt cache (`--prompt-cache-gib`, default 8 GiB; `0` turns it off): kept states hold the global layers' first `at` keys and values, the sliding layers' live window aligned to a 64-key block, the convolution rings, and (with drafts on) the draft head's five context rings (20.3 MiB), taking 32 KiB a token plus up to 223 MiB (about 1.2 GiB at 32k tokens). States are saved at prompt-pass marks and restored into any cache capacity with identical output bits; a follow-up turn prefills the previous reply and the new message from the history mark. Streams take cache memory first and kept states evict before any stream is refused.
- Started by launchd (a router run as a LaunchDaemon or LaunchAgent), the server needs two things:
  - `ProcessType` `Interactive` in the job's plist. Without it, launchd clamps the job and its children to
    background priority, a decode step's wait wakes tens of milliseconds late, and drafted decode drops to about
    15 tok/s from 61.
  - The model directory's files on a disk the job may read. macOS guards external volumes from launchd jobs that
    were not granted access: under a LaunchDaemon, opening a file there blocks without an error, so the server
    loads the rest and then waits, silent and never listening. Copy `mtp.safetensors` rather than linking it to such
    a volume.
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
- `INKLING_PREFILL_ROWS` sets the rows of a prompt chunk at most (a multiple of 64 from 64 to 8,192; default 4,096, or 512 on the
  per-operation path below). The
  scratch is made for that many rows, so a larger chunk takes more of the load limit (2.67 GiB at 4,096 rows, 0.37 GiB at
  512, for the published pack), and a prompt is fed in pieces of that many. Every path below keeps a row's bits
  independent of the chunk it is in.
- On a GPU with tensor units a prompt chunk takes the chunk path: the dense projections and the router as GEMMs (bf16
  operands, f32 sums in a fixed order), attention over 64-key blocks that reads a relative-bias table built just before
  it, and the routed experts through a chain adapted from Sushi that sorts the picks by expert on the GPU. The chain
  takes a layer whose experts are all K2 or K3; any other layer (a K4 expert, say) runs its experts through the per-row
  grouped projections that decode windows use for such a layer, while its dense projections and attention stay on the
  tensor units. A pack that needs this holds the per-row path's buffers for a whole chunk (4.26 GiB of scratch at 4,096
  rows, 0.60 GiB at 512).
- A GPU without tensor units takes the per-operation prompt path of the first release: matvec slabs, a relative-bias table
  and one-row attention, and the per-row expert projections. It runs at that path's speed (about 65-71 tok/s on an M5
  Ultra) and with its scratch. Unless `INKLING_PREFILL_ROWS` is set it takes the first release's 512-row chunks (0.57 GiB
  of scratch; 4.0 GiB at 4,096 rows). The startup log names the path, its rows and its scratch.
- `INKLING_PROMPT_PATH` is a diagnostic: `per_op` runs prompt chunks on the per-operation kernels whatever the GPU, to
  compare the paths or to test that fallback; `rows` (every layer) or `rows:3,7` (those layers) sends layers' experts
  through the per-row projections, in decode windows too, as if the chain could not take them. Unset or `auto` takes what
  the GPU and the pack allow. A cache holds a multiple of 64 positions.
- A chunk's layers go to the GPU in several command buffers, committed as they are encoded and run in order, because the
  GPU kills a command buffer that runs for long (a kill at 15.6 s has been seen). Each buffer holds at most 3,072 rows
  times layers, and always at least one whole layer: a layer at a time from 2,048 rows up (so at 4,096 rows and above it
  is one layer per buffer), six layers at 512. The bits are the same as from one buffer.
- `tf-inkling-run MODEL prefill ids.json [--chunk N] [--layers L] [--repeat R] [--spans] [--drafts]` feeds a prompt
  through the serving prompt path and prints its time, its GPU time by class, the command buffers it used and the
  longest one's GPU time, and a digest of the last row's hidden state (the same at any chunk size) (`--spans` adds a line
  a chunk; `--repeat` runs the loaded layers R times a chunk, so a one-layer pack makes a deep model's command buffers;
  `--drafts` adds the draft head's taps and absorb). It exits non-zero when the GPU times add up to more than the wall
  time. `INKLING_PREFILL_ROWS` sets the chunk of its other modes too.
- `tf-inkling-run MODEL layers ids.json out.bin L [--window] [--chunk N]` and `tf-inkling-run MODEL finals index.txt
  out_dir [--window] [--chunk N]` feed their ids as prompt chunks of N rows (the prefill rows when not given), so they exercise the chunk
  path's arithmetic; `--window` feeds them the way a decode-time client does instead.
- `tf-inkling-run MODEL partition ids.json CUT... [--restore AT]... [--layers L] [--drafts]` feeds a prompt twice, in chunks of the prefill
  rows and in chunks of the lengths given (the prefill rows for what is left), plus one leg per `--restore AT` that saves a prompt state at `AT` and resumes on a fresh poisoned cache, each leg on a scratch and a cache poisoned
  with NaN past the rows in use, and compares the legs bit for bit: every layer's keys and values, the convolution rings,
  every row's last-layer output and expert picks and, on the whole model, the taps, the last row's finals and the kept
  logits (with `--drafts`, the first hold's proposals). It exits non-zero on a difference, on a NaN in a result and on a
  write past a chunk's rows.

## Drafts

The draft head proposes up to 7 tokens a round from the target's features (layers 5 to 41 and the final row); the
target verifies them in one window, so every reply has the same tokens as plain decoding, greedy or sampled with a
seed.
- The head's layer projections are quantized at load to the MLX affine 8-bit form (group 64); the Markov head,
  norms and unembedding stay bf16. It takes 1.9 GiB. `INKLING_DRAFT_BITS=16` keeps the layers bf16 (3.3 GiB).
- `INKLING_DRAFT_CONFIDENCE` (default 0.7) is the floor for a draft's own probability, the top-1 probability of its
  step: a round verifies the first draft, then each next one while its probability is at least the floor. `off`
  verifies every draft a round holds.
- `INKLING_DRAFT_DEPTH` caps the drafts held (1 to 7; unset is 7 with a floor, 4 with `off`); `auto` leaves the depth
  to the lane core's cost rule.
- `tf-inkling-run MODEL greedy ids.json STEPS --drafts` decodes with drafts and prints the rounds and acceptance;
  `tf-inkling-run MODEL dspark TAPS SEQS OUT` replays the head on recorded features, and `tf-inkling-bench dspark`
  times a hold.

Measured on an M5 Ultra (256 GB), served, one stream, 256-token replies over a bench prompt and 12 chat prompts
(about 60 to 1,100 prompt tokens; decode tokens over decode time):

| | |
|---|---|
| Prompt (prefill), 8k-token prompts, drafts on | 2,057-2,090 tok/s (median of three requests on each of three prompts of 8,191-8,193 tokens; about 3.9 s to the first token), up from about 65-71 tok/s on the per-operation kernels |
| Prompt, direct start, 8,192 tokens | 2,119 tok/s (3,867 ms: experts 2,325 ms, dense projections 675 ms, norms 422 ms, attention 298 ms, the rest about 100 ms) |
| Prompt, direct start, 131,008 tokens, drafts on | 1,618 tok/s (80.95 s wall, 78.35 s GPU, max command buffer 216 ms) |
| Resumed 32k follow-up / regenerate / edit / reask (served, prompt cache on) | 0.51 s / 0.20 s / 0.50 s / 2.45 s TTFT (vs 16.7-17.3 s cache off; bit-identical replies, 1.00x decode rate) |
| Decode after 32,704 tokens (direct start) | 56.3 tok/s with drafts; 37.1 tok/s plain |
| Decode after 131,008 tokens (direct start) | 45.7 tok/s with drafts; 31.9 tok/s plain |
| Decode with drafts, three gate prompts | 60.6 tok/s (61.6 before the chunk path) |
| Decode with drafts (default) | 61.4 tok/s (65.2 on the bench prompt) |
| Decode with drafts, every draft verified at depth 4 | 57.1 tok/s |
| Decode, `--no-drafts` | 39.4 tok/s; 37.9 tok/s after an 8,202-token prompt |
| Decode with drafts, sampled (temperature 0.7, no truncation) | 56.8 tok/s |
| Decode, sampled (temperature 0.7; four short prompts with top_p 0.95, top_p 0.95 and top_k 20, min_p 0.05, none) | 44.7 tok/s with drafts; 38.7 tok/s with `--no-drafts` |

Decode steps of up to 16 rows take a decode path: the experts run a rate-grouped EXL3 chain (adapted from Sushi), and
attention, norms and the attention-input matvecs have their own kernels. A step takes about 25 ms of GPU time. Prompt
rows take the chunk path above.

## Fidelity

Teacher-forced on a 62,526-token held-out set against the MXFP8 release as teacher, with the per-operation prompt
kernels (the path a GPU without tensor units still takes); the chunk path was then checked against it:
- **KL:** 0.0325 on the prompt path; 0.0336 with every row after the first 16 fed through the decode path in 16-row
  windows.
- **Comparison:** the same pack's experts in an MLX forward score KL 0.0341. TensorFold's is 0.0016 lower on the
  prompt path (95% CI −0.0035 … −0.0001) and 0.0005 lower on the decode path (95% CI −0.0020 … +0.0011).
- **Top-1 agreement:** 95.5%.
- **Per layer:** hidden states stay within 1.3× the difference between two correct bf16 implementations (1.21× on the
  decode path).
- **Chunk path:** the held-out KL 95% upper bound of TensorFold minus the MLX reference is +0.00073 in chunk mode
  (gate +0.002). The per-layer gate's worst layer is 1.096× the noise floor in chunk mode and 1.062× in window mode
  (limit 1.5×); `layers` and `finals` run in either mode.
- **Decode windows:** a window of up to 16 rows gives the same bits as the same rows decoded one at a time.

## Limits

- One stream; `--parallel` above 1 is refused.
- `--vision` is refused.
- The server does not yet split Inkling's reply channels: a thinking reply's `reasoning_content` starts with the
  raw channel markers (`<|message_model|><|content_thinking|>`). Send
  `chat_template_kwargs {"enable_thinking": false}`, or serve with `--no-thinking`, for replies without thinking.
