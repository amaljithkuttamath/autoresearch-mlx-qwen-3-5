# autoresearch-mlx-qwen-3-5

Tiny Qwen3.5-style hybrid attention experiment running on **Apple M4 Pro (24GB unified memory)** via MLX. The model alternates Gated DeltaNet (linear attention) and full softmax attention layers in a 3:1 pattern. The goal: find out if hybrid attention beats pure full attention at small scale under a fixed 5-minute training budget.

## Architecture

The model uses a Qwen3.5-inspired architecture:
- **Hybrid attention**: 3 DeltaNet layers + 1 full attention layer (pattern: LLLF)
- **Gated DeltaNet**: recurrent linear attention with delta rule, causal conv1d, exponential decay gating
- **Full attention**: gated Q projection (2x Q dim, sigmoid gate on output), QK-norm via RMSNorm, RoPE
- **SwiGLU MLP**: 3 linear layers (gate, up, down)
- **RMSNorm**: (1 + weight) scaling with zero-init
- **No**: value embeddings, residual lambdas, logit capping

Default config: 4 layers, 256 embed dim, 4 heads, 64 head dim, ~21M total params (~4M non-embedding).

**Important constraint**: The recurrent DeltaNet loop is slow at long sequences. On M4 Pro 24GB, DEVICE_BATCH_SIZE=4 gives ~3.8s/step (~78 steps in 5 min). Batch size 16 is too slow (~85s/step). Keep DEVICE_BATCH_SIZE at 4 or lower unless you find a way to speed up the recurrent loop. Memory budget is 24GB unified -- leave headroom for the OS.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git init && git add -A && git commit -m "initial"` if no git repo exists, then `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `prepare.py` -- fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` -- the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, layer pattern, DeltaNet parameters, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the layer pattern. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. MLX uses unified memory. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Speed matters.** The recurrent DeltaNet is the bottleneck. More optimizer steps generally means better results. If you find a way to speed up the DeltaNet (chunked processing, reduced head dim, fewer DeltaNet layers), that frees up more steps. Trade-off: fewer DeltaNet layers means less linear attention benefit, but more steps.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Experiment ideas (starting points)

These are suggestions, not prescriptions. The agent should generate its own ideas based on results.

**Critical first experiments (run these early):**
- **Pure attention baseline (FFFF)**: Change LAYER_PATTERN to "FFFF" to get a pure full-attention baseline. Without this, you cannot know if DeltaNet is helping or hurting. This should be experiment #2 after the hybrid baseline.
- **Pure DeltaNet (LLLL)**: All linear attention, no full attention. Tests the other extreme.

**Architecture tuning:**
- **Layer pattern**: Try LLFF, LF, LLLF, LLLLF to find the best ratio
- **DeltaNet head dim**: Reduce from 64 to 32 -- faster recurrence, allows more steps
- **DeltaNet heads**: Try 2 instead of 4
- **Conv kernel size**: Try 2 or 8 instead of 4
- **Model dim**: Try 384 or 512 (more capacity, fewer steps -- trade-off)
- **MLP expansion**: Try 2.0 or 4.0 instead of 3.5

**Optimizer tuning:**
- **Learning rates**: The defaults are from the original nanochat, may not be optimal for this architecture
- **Warmup**: Try WARMUP_RATIO=0.05, the DeltaNet recurrent state might benefit from warmup
- **Batch size**: Try DEVICE_BATCH_SIZE=2 for even more steps

**Speed optimizations:**
- **Smaller DeltaNet**: head_dim=32, heads=2 makes the recurrence 4x cheaper
- **Fewer DeltaNet layers**: LLLF -> LF or LLF, then use the extra speed for more steps or larger model

**Note on FINAL_EVAL_BATCH_SIZE**: Currently set to 4 (same as DEVICE_BATCH_SIZE). If eval results seem noisy between runs, this may need increasing, but it will slow down the eval phase.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          X.XXXXXX
training_seconds: 300.1
total_seconds:    350.0
peak_vram_mb:     XXXXX.X
mfu_percent:      0.00
total_tokens_M:   XX.X
num_steps:        XX
num_params_M:     21.0
depth:            4
```

Extract the key metric from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) -- use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) -- use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything -- do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (do not commit results.tsv, leave it untracked)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~6 minutes total (5 min training + ~1 min compile/eval). With the recurrent DeltaNet, some experiments may take longer if batch size is increased. If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you, period.

On M4 Pro 24GB, each experiment takes ~6 minutes, so you can run ~10/hour, ~50 overnight. The user wakes up to a results.tsv full of experiments.
