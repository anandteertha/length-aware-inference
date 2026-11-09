import importlib.metadata
import platform
import time
from collections import Counter


def environment():
    import cpuinfo
    import onnxruntime as ort
    import psutil

    cpu = cpuinfo.get_cpu_info()
    packages = {}
    for name in ("torch", "transformers", "datasets", "onnx", "onnxruntime", "numpy", "etdl"):
        packages[name] = importlib.metadata.version(name)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu.get("brand_raw", platform.processor()),
        "cpu_flags": cpu.get("flags", []),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "total_memory_bytes": psutil.virtual_memory().total,
        "available_memory_bytes": psutil.virtual_memory().available,
        "available_providers": ort.get_available_providers(),
        "packages": packages,
    }


def session(path, threads=1, optimized_path=None, profile_prefix=None):
    import onnxruntime as ort

    if threads < 1:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if optimized_path is not None:
        options.optimized_model_filepath = str(optimized_path)
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    start = time.perf_counter_ns()
    result = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return result, (time.perf_counter_ns() - start) / 1e6


def graph_operators(path):
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    return dict(
        sorted(
            Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in graph.node).items()
        )
    )
