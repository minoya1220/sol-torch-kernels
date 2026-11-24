# sol-torch-kernels

Speed-of-light PyTorch op kernels for NVIDIA Blackwell (sm120), written in CuTe DSL.

Status: the evaluation harness and the first op (softmax forward and backward) are being built. See [docs/eval-design.md](docs/eval-design.md) for how kernels are judged and [AGENTS.md](AGENTS.md) for how the repo is worked on.

## Layout

- `src/sol_kernels/`: the kernels.
- `src/sol_eval/`: the harness that checks and times them.
- `eval/exploits/`: cheats the harness must reject, each with the defence that catches it.

## Setup

```
uv sync
```

Running kernels needs an NVIDIA GPU with compute capability 12.0 (sm120).
