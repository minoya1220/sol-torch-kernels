# Working in this repo

Understanding and legibility beat volume. The owner must be able to read and understand every line of the harness, because the harness is what decides whether an AI-written kernel is correct and fast. Code nobody reads carefully is where reward hacks hide.

The reference point is https://github.com/minoya1220/kernels: softmax forward and backward, a CUDA-graph benchmark with L2 rotation, and a persisted tuner in a few small, readable files. Aim for that clarity and intentionality.

## Rules

1. **Every line earns its place.** Before adding code, say what concrete problem it solves today. If the answer is "a future op might need it", don't write it.
2. **Defend against demonstrated cheats, not imagined ones.** The design's threat model is a map, not a build list. A defence goes in when an exploit in `eval/exploits/` shows it is needed, or when the defence is a one-line rule that costs nothing. The exploit and its defence land together.
3. **Simple, not short.** Line count is not the goal and must not be gamed: no dense one-liners, clever tricks or cramming to look small. Write the plainest code that does the job, so a reader can follow it top to bottom. Bloat is the thing to avoid: code, layers, options and tests that exist without a present reason. Keep PRs small enough to review carefully in one sitting.
4. **Plain code.** Functions over classes, one obvious place for each thing, no frameworks, plugin systems, registries or config knobs nobody sets yet. An abstraction needs at least two real users. Prefer torch and the CuTe DSL over code that reimplements them.
5. **Few, sharp tests.** Each test guards one property you can name. No snippet explosions: one honest case and the exploit that matters beat eighty variations. The exploit corpus is the integrity test suite.
6. **Short docs.** The design doc says why; the code says how. Comments explain why something is non-obvious, never what the next line does. Don't restate the design in docstrings or PR descriptions.
7. **Think before building.** When the design is ambiguous or a rule looks wrong or expensive, stop and ask in the PR instead of building around it. A short question is better than a large guess.
8. **Never weaken a check to make something pass**, and never merge your own PR. Harness, test and eval changes need the owner's review and approval.
