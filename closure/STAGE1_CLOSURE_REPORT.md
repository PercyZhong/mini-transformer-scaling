# Stage 1 Checkpoint Closure Report

## Outcome

- Formal runs completed: 9/9
- Loadable non-empty best checkpoints: 9/9
- Loadable non-empty final checkpoints: 9/9
- Independent strict reload and `ROMEO:` generation tests: 9/9
- Missing best checkpoints: 0
- Retraining triggered: no

## Repository record

- GitHub repository: `https://github.com/PercyZhong/mini-transformer-scaling.git`
- Original experiment commit: `bd74aa17ffad8d2db7d0aa9214199a83f2c401cc`
- Closure implementation commit: `359a820e282a0f7af4c270e10488d2980bab5ff8`
- Git LFS used: no
- Checkpoints committed to ordinary Git history: no
- Checkpoint storage: final closure ZIP only

## Validation method

Each `best.pt` was loaded in a newly spawned Python process on CPU. The process rebuilt
`MiniTransformerLM` from the saved model configuration, loaded the state dictionary with
`strict=True`, reconstructed the saved vocabulary, selected `eval()` through the generation
helper, fixed seed 42, and generated 100 characters from `ROMEO:`. SHA256 was calculated from
the exact archived bytes. The inventory also confirms step, optimizer, AMP scaler, batch
iterator, RNG, configuration, vocabulary, and model state required for recovery. The learning
rate schedule is stateless and derived from step/configuration, so there is no scheduler object.

## Notes

The existing experimental metrics and summaries were not changed. Both `best.pt` and
`final.pt` were retained because storage was sufficient. The archive SHA256 is stored in the
ZIP sidecar file because a ZIP cannot contain its own final hash without changing that hash.
