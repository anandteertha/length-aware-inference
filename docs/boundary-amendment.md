# Timing-quality amendment: interrupted diagnostic

Recorded November 10, 2026, while the original `artifacts/boundaries` run was
still executing, before its final summaries or per-length performance ratios
were inspected.

After a date/context update, process telemetry showed a long interruption inside
block 2, transformer family, capacity 16. Its observation duration exceeded
32,000 seconds, and adjacent telemetry samples were separated by over 21,000
seconds. The exact cause was not instrumented; a stable before/after power-scheme
and AC snapshot does not establish uninterrupted execution.

Preserve every process and request from the original run, with no exclusions.
Archive it separately as `results/boundaries-interrupted`. Identify any process
whose first CPU sample arrives after ten seconds or whose adjacent one-second
CPU samples have a gap greater than ten seconds. This is an instrumentation
quality flag, not a latency threshold or an outcome-based outlier rule.

Once that run finishes, repeat the same frozen synthetic design in the new
`artifacts/boundaries-repeat` directory, without overlapping timing workers.
Use the unchanged measurement code, graph hashes, lengths, patterns, thread count,
five blocks, eight repeats, two warm-ups, two-second rests, and seed 2041. Preserve
all replication outcomes regardless of whether they favor any graph.

The consolidated report must disclose both runs. Use the fresh replication for
the main mechanism plots only if it has no flagged telemetry gaps, passes output
checks, and matches recorded before/after power conditions. Report the original
run's results separately and do not silently combine the two into ten independent
uninterrupted blocks. If the replication also has a long interruption, report the
remaining limitation rather than repeating until a favorable result appears.
