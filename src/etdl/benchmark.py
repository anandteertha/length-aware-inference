import random
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np

from etdl.checks import verified_dynamic
from etdl.data import encode, load_calibration, load_tokenizer
from etdl.io import sha256, write_json
from etdl.runtime import environment, session


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latency measurements must be nonempty, finite, and nonnegative")
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95, method="linear")),
        "mean_ms": float(np.mean(values)),
    }


@contextmanager
def sampled_memory(interval=0.01):
    import psutil

    process = psutil.Process()
    initial = process.memory_info().rss
    measurement = {
        "rss_before_bytes": initial,
        "sampled_peak_rss_bytes": initial,
        "sampling_interval_ms": interval * 1000,
    }
    stop = threading.Event()

    def observe():
        while not stop.is_set():
            measurement["sampled_peak_rss_bytes"] = max(
                measurement["sampled_peak_rss_bytes"], process.memory_info().rss
            )
            stop.wait(interval)

    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    try:
        yield measurement
    finally:
        stop.set()
        worker.join()
        info = process.memory_info()
        measurement["rss_after_bytes"] = info.rss
        measurement["sampled_peak_rss_bytes"] = max(measurement["sampled_peak_rss_bytes"], info.rss)
        measurement["incremental_sampled_peak_bytes"] = max(
            0, measurement["sampled_peak_rss_bytes"] - initial
        )
        # Windows exposes the lifetime OS high-water mark. It includes imports before sampling.
        measurement["process_lifetime_peak_rss_bytes"] = getattr(info, "peak_wset", None)


def benchmark(config, limit, threads, repeats, warmup):
    if limit <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("limit/repeats must be positive; warmup must be nonnegative")
    calibration_path = config.artifact_dir / "data/calibration.json"
    rows = load_calibration(config, limit)
    path = verified_dynamic(config, rows)
    tokenizer = load_tokenizer(config)
    # Record environment outside timing; this worker does not instantiate a PyTorch model.
    machine = environment()
    with sampled_memory() as memory:
        runtime, startup_ms = session(path, threads)
        for i in range(warmup):
            runtime.run(
                ["logits"], encode(tokenizer, rows[i % len(rows)]["text"], config.max_length)
            )
        samples = []
        rng = random.Random(config.seed)
        for repeat in range(repeats):
            order = list(range(len(rows)))
            rng.shuffle(order)
            for index in order:
                row = rows[index]
                start = time.perf_counter_ns()
                inputs = encode(tokenizer, row["text"], config.max_length)
                tokenized = time.perf_counter_ns()
                logits = runtime.run(["logits"], inputs)[0]
                completed = time.perf_counter_ns()
                samples.append(
                    {
                        "repeat": repeat,
                        "idx": row["idx"],
                        "tokens": row["tokens"],
                        "label": row["label"],
                        "prediction": int(logits.argmax(-1)[0]),
                        "tokenization_ms": (tokenized - start) / 1e6,
                        "inference_ms": (completed - tokenized) / 1e6,
                        "end_to_end_ms": (completed - start) / 1e6,
                    }
                )
    # ORT instrumentation runs separately so tracing cannot inflate timed requests.
    traces = config.artifact_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    del runtime
    profiled, _ = session(path, threads, profile_prefix=traces / f"dynamic-t{threads}")
    profiled.run(["logits"], encode(tokenizer, rows[0]["text"], config.max_length))
    trace_path = profiled.end_profiling()
    summaries = {
        name: latency_summary([sample[name] for sample in samples])
        for name in ("tokenization_ms", "inference_ms", "end_to_end_ms")
    }
    by_length = {}
    for length in sorted({sample["tokens"] for sample in samples}):
        by_length[str(length)] = latency_summary(
            [sample["end_to_end_ms"] for sample in samples if sample["tokens"] == length]
        )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "optimized_dynamic_onnx",
        "split": "train",
        "purpose": "calibration_smoke_benchmark",
        "examples": len(rows),
        "repeats": repeats,
        "warmup_requests": warmup,
        "threads": threads,
        "batch_size": 1,
        "provider": "CPUExecutionProvider",
        "optimization_level": "ORT_ENABLE_ALL",
        "session_initialization_ms": startup_ms,
        "accuracy": sum(s["prediction"] == s["label"] for s in samples) / len(samples),
        "latency": summaries,
        "end_to_end_by_token_length": by_length,
        "memory": memory,
        "model_size_bytes": path.stat().st_size,
        "model_sha256": sha256(path),
        "calibration_sha256": sha256(calibration_path),
        "environment": machine,
        "ort_trace": trace_path,
        "samples": samples,
        "limitations": [
            "Training-data calibration measurements are not held-out accuracy or speed results.",
            "End-to-end covers tokenization and synchronous inference, excluding startup and I/O.",
            "Session initialization is measured with the model potentially in the OS file cache.",
            "Sampled process RSS includes the Python runtime and tokenizer; it is not tensor memory.",
            "Repeated requests are dependent observations; no confidence interval is claimed.",
        ],
    }
    output = config.artifact_dir / f"reports/benchmark-t{threads}.json"
    write_json(output, report)
    return {
        "path": str(output),
        "examples": len(rows),
        "accuracy": report["accuracy"],
        "latency": summaries,
        "memory": memory,
    }
