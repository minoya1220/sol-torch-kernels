# sol-torch-kernels

Speed-of-light PyTorch op kernels for NVIDIA Blackwell (sm120), written in CuTe DSL.

Status: no kernels yet. The evaluation harness is designed first; see [docs/eval-design.md](docs/eval-design.md).

## Layout

- `src/sol_kernels/`: the kernel library.
- `src/sol_eval/`: the test and benchmark harness (protected path).
- `docs/eval-design.md`: harness design (protected path).

## Setup

```
uv sync
```

Requires an NVIDIA GPU with compute capability 12.0 (sm120) to run kernels.
