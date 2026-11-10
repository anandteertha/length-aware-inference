# Run the CPU classifier demonstration

From this completed workspace, run:

```powershell
uv sync --locked
uv run python tools/demo.py
uv run python tools/demo.py --text "Great service!" "This was a terrible movie."
uv run etdl predict "A thoughtful film with excellent performances."
```

The demo loads the original frozen policy once, then reuses its sessions for all
sentences. It prints the predicted label, FP32 logits, token count, route, and
policy digest. The current frozen policy is dynamic-only ONNX at four threads.
The two built-in sentences are functional examples, not an accuracy evaluation;
their observed timings are not benchmark estimates.

To save the run locally:

```powershell
uv run python tools/demo.py --output artifacts/reports/project-demo.json
```

This output includes the supplied text. The classifier's separate routing log
records decisions without input text. Keep saved custom-text demo outputs local
when they contain private information.

## Starting from a fresh checkout

Model binaries and downloaded datasets are not committed. Run the setup,
milestone, variants, and study commands in the [README](../README.md) first, in a
fresh artifact directory. The demo prefers `confirmation/frozen-policy.json`
when available and otherwise uses `study-t1/selection.json`; `--policy PATH`
selects another verified policy. The `--config PATH` option selects a dedicated
artifact directory through a configuration file.

A newly calibrated policy describes that new machine and run. The original
archived policy checks CPU/runtime identity and model hashes and deliberately
does not serve as a portable deployment recommendation. No validation evaluation
is required to run a functional demonstration, and the diagnostic does not change
the existing policy.

## Repeating the length diagnostic

After the verified follow-up graph families exist locally:

```powershell
uv run etdl boundaries --name boundaries
# Resume an interrupted run without changing its code, inputs, or settings:
uv run etdl boundaries --name boundaries --resume
uv run python tools/archive_boundaries.py --diagnostic artifacts/boundaries --output results/my-boundary-run
```

Use a new name for a new run; the completed `boundaries` directory is preserved.
The [protocol](boundary-protocol.md) specifies five randomized process blocks,
fixed inputs, output checks, power-state checks, load logging, and interpretation
limits. The diagnostic uses synthetic token IDs and excludes text tokenization.
