# autoresearch-mlx-qwen-3-5

Tiny Qwen3.5-style hybrid attention experiment for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) framework, running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## What is this?

A from-scratch implementation of Qwen3.5's hybrid attention architecture (Gated DeltaNet + full softmax attention) scaled down to ~8M params, plugged into the autoresearch autonomous experimentation loop. The question: does hybrid linear/full attention beat pure full attention at small scale under a fixed 5-minute training budget?

Based on [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (data pipeline and eval), with the model architecture replaced.

## Architecture

- 4 layers: 3 Gated DeltaNet (linear attention) + 1 full softmax attention (pattern: LLLF)
- SwiGLU MLP (gate/up/down, 3.5x expansion)
- RMSNorm with (1+w) scaling, zero-init
- Gated Q projection on full attention layers
- Chunk-wise delta rule with causal conv1d on linear attention layers
- QK-norm (RMSNorm for full attn, L2 for DeltaNet)
- ~8.3M params at default config (vocab_size=8192)

## Hardware

Developed and tested on M4 Pro 24GB. DEVICE_BATCH_SIZE=4 gives ~345ms/step (~820 steps in 5 min).

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py    # one-time data download + tokenizer
uv run train.py      # ~6 min (5 min training + compile/eval)
```

Then point Claude Code (or any agent) at `program.md` and let it run experiments autonomously.

## Three files that matter

- **prepare.py** -- data prep, tokenizer, dataloader, evaluation. Do not modify.
- **train.py** -- model, optimizer, training loop. The agent edits this.
- **program.md** -- agent instructions. Point your agent here.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) -- autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- MLX port
- [Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch) -- Qwen3.5 from-scratch reference
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
