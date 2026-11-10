"""Fresh training-only experiments with per-thread selection and transformer fusions."""

import logging
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from etdl.benchmark import latency_summary, sampled_memory
from etdl.checks import compare_logits, verified_dynamic
from etdl.data import DATASET_ID, encode, load_calibration, load_tokenizer, model_source
from etdl.io import read_json, sha256, write_json
from etdl.runtime import environment, graph_operators, session
from etdl.selection import learn_routes, route_name, select_configuration, subsets
from etdl.study import hardware_key
from etdl.variants import pad_inputs

LOGGER = logging.getLogger(__name__)


def source_graphs(config):
    from etdl.variants import verified_variants

    rows = load_calibration(config)
    return {"dynamic": verified_dynamic(config, rows), **verified_variants(config, rows)}


def partition_rows(rows, excluded, sizes=(256, 128, 128)):
    if any(size < 1 for size in sizes):
        raise ValueError("Partition sizes must be positive")
    selected, seen = [], set(excluded)
    for row in rows:
        if row["idx"] not in seen:
            selected.append(row)
            seen.add(row["idx"])
        if len(selected) == sum(sizes):
            break
    if len(selected) != sum(sizes):
        raise ValueError("Insufficient unused training rows")
    result, offset = {}, 0
    for role, size in zip(("fit", "confirmation", "audit"), sizes, strict=True):
        result[role] = selected[offset : offset + size]
        offset += size
    return result


def prepare(config, directory, seed=2037, sizes=(256, 128, 128)):
    from datasets import load_dataset

    manifest_path = directory / "partitions.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["seed"] != seed or manifest["sizes"] != list(sizes):
            raise ValueError("Existing follow-up partitions use different settings")
        if manifest["calibration_sha256"] != sha256(
            config.artifact_dir / "data/calibration.json"
        ) or manifest["previous_confirmation_sha256"] != sha256(
            config.artifact_dir / "confirmation/training.json"
        ):
            raise ValueError("Prior training exclusions changed after partitioning")
        for role, digest in manifest["workload_hashes"].items():
            if sha256(directory / f"{role}.json") != digest:
                raise ValueError("Reserved follow-up data changed")
        return manifest
    calibration_path = config.artifact_dir / "data/calibration.json"
    calibration = read_json(calibration_path)
    previous = config.artifact_dir / "confirmation/training.json"
    if not previous.exists():
        raise ValueError("The previous confirmation workload is required for exclusion")
    excluded = {row["idx"] for row in calibration["rows"]}
    excluded.update(row["idx"] for row in read_json(previous)["rows"])
    dataset = load_dataset(DATASET_ID, "sst2", split="train", revision=calibration["revision"])
    dataset = dataset.shuffle(seed=seed)
    selected = partition_rows(dataset, excluded, sizes)
    tokenizer = load_tokenizer(config)
    directory.mkdir(parents=True, exist_ok=True)
    for role, rows in selected.items():
        encoded = []
        for row in rows:
            inputs = encode(tokenizer, row["sentence"], config.max_length)
            encoded.append(
                {
                    "idx": int(row["idx"]),
                    "text": row["sentence"],
                    "label": int(row["label"]),
                    "tokens": int(inputs["input_ids"].shape[1]),
                }
            )
        write_json(
            directory / f"{role}.json",
            {
                "split": "train",
                "role": role,
                "seed": seed,
                "dataset_id": DATASET_ID,
                "revision": calibration["revision"],
                "rows": encoded,
                "model_source": model_source(config),
            },
        )
    manifest = {
        "split": "train",
        "seed": seed,
        "sizes": list(sizes),
        "excluded_training_ids": sorted(excluded),
        "model_source": model_source(config),
        "calibration_sha256": sha256(calibration_path),
        "previous_confirmation_sha256": sha256(previous),
        "workload_hashes": {role: sha256(directory / f"{role}.json") for role in selected},
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_graph(config, name, candidate):
    """Check original examples and mask patterns before a new graph enters profiling."""
    baseline, _ = session(verified_dynamic(config), 1, allow_spinning=False)
    runtime, _ = session(candidate, 1, allow_spinning=False)
    tokenizer = load_tokenizer(config)
    capacity = int(name.split("-")[1]) if name != "dynamic" else None
    cases = []

    def compare(case, inputs):
        feed = pad_inputs(inputs, capacity, tokenizer.pad_token_id) if capacity else inputs
        actual = runtime.run(["logits"], feed)[0]
        expected = baseline.run(["logits"], inputs)[0]
        cases.append({"case": case, **compare_logits(expected, actual, config.atol, config.rtol)})

    eligible = [
        row for row in load_calibration(config) if capacity is None or row["tokens"] <= capacity
    ]
    for row in eligible[:16]:
        compare(f"prior-train:{row['idx']}", encode(tokenizer, row["text"], config.max_length))
    sizes = (
        sorted({2, capacity // 2, capacity - 1, capacity})
        if capacity
        else [2, 8, 16, 17, 32, 64, 128, 140, 256, 512]
    )
    for size in sizes:
        if size > config.max_length:
            continue
        for mode in ("full", "right_pad", "internal"):
            ids = np.full((1, size), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
            ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            mask = np.ones_like(ids)
            if mode == "right_pad" and size > 2:
                end = max(2, size // 2)
                ids[0, end - 1] = tokenizer.sep_token_id
                ids[0, end:] = tokenizer.pad_token_id
                mask[0, end:] = 0
            if mode == "internal" and size > 2:
                mask[0, 1:-1:2] = 0
                ids[mask == 0] = tokenizer.pad_token_id
            compare(f"mask:{mode}:{size}", {"input_ids": ids, "attention_mask": mask})
    return {
        "passed": all(case["passed"] for case in cases),
        "case_count": len(cases),
        "changed_predictions": sum(not case["labels_match"] for case in cases),
        "cases": cases,
    }


def build_family(config, directory, family):
    """Keep source models intact; optimized graphs live in a separate family directory."""
    import onnx

    source = source_graphs(config)
    registry_path = directory / f"registry-{family}.json"
    if registry_path.exists():
        return load_registry(config, registry_path)
    graph_dir = directory / "models" / family
    graph_dir.mkdir(parents=True, exist_ok=True)
    graphs = {}
    recipe = None
    for name, original in source.items():
        try:
            if family == "standard":
                path = original
                checks = {
                    "passed": True,
                    "scope": "Original checked graphs; all new timed outputs are separately verified.",
                }
                statistics = {}
            elif family == "transformer":
                from onnxruntime.transformers.fusion_options import FusionOptions
                from onnxruntime.transformers.optimizer import optimize_model

                path = graph_dir / f"{name}.onnx"
                recipe = {
                    "model_type": "bert",
                    "num_heads": 12,
                    "hidden_size": 768,
                    "opt_level": 1,
                    "use_gpu": False,
                    "raw_attention_mask": True,
                    "enable_gelu_approximation": False,
                    "precision": "fp32",
                }
                options = FusionOptions("bert")
                options.use_raw_attention_mask(True)
                options.enable_gelu_approximation = False
                optimized = optimize_model(
                    str(original),
                    model_type="bert",
                    num_heads=12,
                    hidden_size=768,
                    optimization_options=options,
                    opt_level=1,
                    use_gpu=False,
                )
                statistics = optimized.get_fused_operator_statistics()
                optimized.save_model_to_file(str(path), use_external_data_format=False)
                del optimized
                model = onnx.load(str(path))
                if any(
                    tensor.data_type in {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16}
                    for tensor in model.graph.initializer
                ):
                    raise ValueError("Optimizer changed floating-point precision")
                onnx.checker.check_model(model)
                del model
                checks = verify_graph(config, name, path)
                write_json(graph_dir / f"checks-{name}.json", checks)
            else:
                raise ValueError("Unknown graph family")
            graphs[name] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "source_sha256": sha256(original),
                "passed": checks["passed"],
                "verification": {key: value for key, value in checks.items() if key != "cases"},
                "fusion_statistics": statistics,
                "operators": graph_operators(path),
            }
            print(f"{family}/{name}: verified={checks['passed']}", flush=True)
        except Exception as error:
            LOGGER.exception("Graph-family construction failed for %s/%s", family, name)
            graphs[name] = {
                "passed": False,
                "source_sha256": sha256(original),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"{family}/{name}: unavailable ({type(error).__name__})", flush=True)
    registry = {
        "family": family,
        "model_source": model_source(config),
        "hardware_key": hardware_key(environment()),
        "recipe": recipe,
        "precision": "fp32",
        "calibration_sha256": sha256(config.artifact_dir / "data/calibration.json"),
        "graphs": graphs,
        "available": bool(graphs.get("dynamic", {}).get("passed")),
    }
    write_json(registry_path, registry)
    return registry


def load_registry(config, path):
    registry = read_json(path)
    if registry["model_source"] != model_source(config) or registry["precision"] != "fp32":
        raise ValueError("Follow-up registry belongs to another model or precision")
    if registry["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
        raise ValueError("Follow-up source calibration changed")
    for graph in registry["graphs"].values():
        if graph["passed"] and sha256(Path(graph["path"])) != graph["sha256"]:
            raise ValueError("Registered follow-up model changed")
    return registry


def benchmark(config, request_path, output):
    request = read_json(request_path)
    registry_path, workload_path, reference_path = map(
        Path, (request["registry"], request["workload"], request["references"])
    )
    for path, field in (
        (registry_path, "registry_sha256"),
        (workload_path, "workload_sha256"),
        (reference_path, "references_sha256"),
    ):
        if sha256(path) != request[field]:
            raise ValueError("Follow-up measurement inputs changed")
    registry = load_registry(config, registry_path)
    machine = environment()
    if hardware_key(machine) != registry["hardware_key"]:
        raise ValueError("Follow-up runtime or hardware changed")
    workload, reference = read_json(workload_path), read_json(reference_path)
    if workload["split"] != "train" or workload["role"] != request["role"]:
        raise ValueError("Follow-up only permits its designated training partition")
    if (
        reference["workload_sha256"] != sha256(workload_path)
        or reference["model_source"] != registry["model_source"]
    ):
        raise ValueError("Follow-up references differ")
    if request["role"] in {"confirmation", "audit"}:
        if "freeze" not in request or "freeze_sha256" not in request:
            raise ValueError("Reserved audit requires a frozen confirmation choice")
        frozen_path = Path(request["freeze"])
        if sha256(frozen_path) != request["freeze_sha256"]:
            raise ValueError("Audit policy changed after freezing")
        frozen = read_json(frozen_path)
        expected_stage = (
            "fit_policies_frozen" if request["role"] == "confirmation" else "audit_choice_frozen"
        )
        if frozen["stage"] != expected_stage or frozen["workload_hashes"][
            request["role"]
        ] != sha256(workload_path):
            raise ValueError("Follow-up partition was not frozen for this stage")
        condition = frozen["conditions"][request["condition_id"]]
        if any(
            condition[key] != request[key]
            for key in ("models", "routing", "threads", "registry_sha256")
        ):
            raise ValueError("Follow-up condition changed after freezing")
    graphs = registry["graphs"]
    names = request["models"]
    if any(not graphs.get(name, {}).get("passed") for name in names):
        raise ValueError("Cannot execute an unverified graph")
    if request["kind"] != "profile" and ("dynamic" not in names or len(names) > 3):
        raise ValueError("Routed configurations require dynamic fallback and at most two variants")
    tokenizer = load_tokenizer(config)
    rows = workload["rows"]
    if request["kind"] == "profile" and names[0] != "dynamic":
        rows = [row for row in rows if row["tokens"] <= int(names[0].split("-")[1])]
    if not rows:
        raise ValueError("No eligible follow-up inputs")
    expected = reference["logits"]
    if any(str(row["idx"]) not in expected for row in rows):
        raise ValueError("Incomplete follow-up reference coverage")
    started = time.perf_counter_ns()
    with sampled_memory() as memory:
        runtimes, initialization = {}, {}
        for name in names:
            runtimes[name], initialization[name] = session(
                Path(graphs[name]["path"]), request["threads"], allow_spinning=False
            )
            size = config.max_length if name == "dynamic" else int(name.split("-")[1])
            ids = np.full((1, size), tokenizer.convert_tokens_to_ids("good"), dtype=np.int64)
            ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            runtimes[name].run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})

        def execute(row, repeat):
            start = time.perf_counter_ns()
            feed = encode(tokenizer, row["text"], config.max_length)
            tokenized = time.perf_counter_ns()
            length = feed["input_ids"].shape[1]
            name = (
                names[0]
                if request["kind"] == "profile"
                else route_name(length, request["routing"], runtimes)
            )
            if name != "dynamic":
                feed = pad_inputs(feed, int(name.split("-")[1]), tokenizer.pad_token_id)
            routed = time.perf_counter_ns()
            logits = runtimes[name].run(["logits"], feed)[0]
            finished = time.perf_counter_ns()
            check = compare_logits(expected[str(row["idx"])], logits, config.atol, config.rtol)
            return {
                "idx": row["idx"],
                "tokens": length,
                "repeat": repeat,
                "route": name,
                "label": row["label"],
                "prediction": int(logits.argmax(-1)[0]),
                "tokenization_ms": (tokenized - start) / 1e6,
                "routing_padding_ms": (routed - tokenized) / 1e6,
                "inference_ms": (finished - routed) / 1e6,
                "end_to_end_ms": (finished - start) / 1e6,
                **check,
            }

        for index in range(request["warmup"]):
            if not execute(rows[index % len(rows)], -1)["passed"]:
                raise ValueError("Follow-up warm-up output check failed")
        ready_ms = (time.perf_counter_ns() - started) / 1e6
        samples = []
        for repeat in range(request["repeats"]):
            ordered = list(rows)
            random.Random(request["order_seed"] + repeat).shuffle(ordered)
            samples.extend(execute(row, repeat) for row in ordered)
    report = {
        "family": registry["family"],
        "threads": request["threads"],
        "kind": request["kind"],
        "role": request["role"],
        "split": "train",
        "models": names,
        "variants": [name for name in names if name != "dynamic"],
        "samples": samples,
        "routing": request.get("routing", {}),
        "examples": len(rows),
        "method": request.get("method", request["kind"]),
        "block": request.get("block", 0),
        "passed": all(sample["passed"] for sample in samples),
        "changed_predictions": sum(not sample["labels_match"] for sample in samples),
        "accuracy": sum(sample["prediction"] == sample["label"] for sample in samples)
        / len(samples),
        "latency": {
            name: latency_summary([sample[name] for sample in samples])
            for name in ("tokenization_ms", "routing_padding_ms", "inference_ms", "end_to_end_ms")
        },
        "memory": memory,
        "memory_for_budget_bytes": max(
            memory["sampled_peak_rss_bytes"], memory["process_lifetime_peak_rss_bytes"] or 0
        ),
        "session_initialization_ms": initialization,
        "sessions_ready_ms": ready_ms,
        "model_size_bytes": sum(Path(graphs[name]["path"]).stat().st_size for name in names),
        "request_sha256": sha256(request_path),
        "registry_sha256": sha256(registry_path),
        "workload_sha256": sha256(workload_path),
        "references_sha256": sha256(reference_path),
        "runtime_settings": {"allow_spinning": False},
        "environment": machine,
    }
    write_json(output, report)
    print(
        f"{registry['family']} t{request['threads']} {request['role']}: p95={report['latency']['end_to_end_ms']['p95_ms']:.3f} ms",
        flush=True,
    )


def pick_candidate(groups, budget_bytes):
    eligible = [
        group
        for group in groups
        if group["runs"] >= 3
        and group["passed"]
        and group["changed_predictions"] == 0
        and group["max_peak_bytes"] <= budget_bytes
    ]
    if not eligible:
        raise ValueError("No confirmed follow-up candidate meets correctness and memory limits")
    return min(
        eligible,
        key=lambda group: (
            group["median_run_p95_ms"],
            group["max_peak_bytes"],
            len(group["variants"]),
            group["family"] != "standard",
            group["method"] != "dynamic",
        ),
    )


def execute(
    config,
    config_path,
    directory,
    resume=False,
    threads=(1, 2, 4),
    blocks=3,
    warmup=10,
    budget_mib=1536,
    seed=2037,
    sizes=(256, 128, 128),
):
    """Fit policies per setting, confirm in fresh blocks, freeze, then audit new rows."""
    from etdl.comparison import summarize_runs

    if warmup < 0 or budget_mib <= 0:
        raise ValueError("Invalid follow-up warm-up or memory budget")
    if not threads or min(threads) < 1 or len(set(threads)) != len(threads) or blocks < 3:
        raise ValueError("Use unique positive thread counts and at least three confirmation blocks")
    plan_path = directory / "experiment.json"
    if plan_path.exists() and not resume:
        raise ValueError("Follow-up exists; resume the same immutable plan or use a new name")
    partitions = prepare(config, directory, seed, sizes)
    registries = {
        family: build_family(config, directory, family) for family in ("standard", "transformer")
    }
    available = [family for family, registry in registries.items() if registry["available"]]
    if "standard" not in available:
        raise ValueError("The checked standard fallback family is unavailable")
    registry_paths = {family: directory / f"registry-{family}.json" for family in available}
    settings = {
        "threads": list(threads),
        "blocks": blocks,
        "warmup": warmup,
        "budget_mib": budget_mib,
        "seed": seed,
        "sizes": list(sizes),
    }
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["settings"] != settings:
            raise ValueError("Resume settings differ from the follow-up plan")
    else:
        plan = {
            "stage": "planned",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "settings": settings,
            "workload_hashes": partitions["workload_hashes"],
            "registry_hashes": {family: sha256(path) for family, path in registry_paths.items()},
            "model_source": model_source(config),
            "hardware_key": hardware_key(environment()),
            "measurement_code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        }
        write_json(plan_path, plan)
    if plan["workload_hashes"] != partitions["workload_hashes"]:
        raise ValueError("Follow-up partition manifest changed")
    if plan["registry_hashes"] != {family: sha256(path) for family, path in registry_paths.items()}:
        raise ValueError("Follow-up graph registry changed after planning")
    if plan["hardware_key"] != hardware_key(environment()):
        raise ValueError("Follow-up machine/runtime changed")
    measurements = directory / "measurements"
    measurements.mkdir(exist_ok=True)
    worker = [sys.executable, "-m", "etdl.worker", "--config", str(config_path.resolve())]

    def reference(role):
        path = directory / f"references-{role}.json"
        if not path.exists():
            print(f"Preparing {role} references", flush=True)
            subprocess.run(
                worker
                + [
                    "reference",
                    "--workload",
                    str(directory / f"{role}.json"),
                    "--output",
                    str(path),
                ],
                check=True,
            )
        payload = read_json(path)
        if (
            payload["workload_sha256"] != plan["workload_hashes"][role]
            or payload["model_source"] != plan["model_source"]
        ):
            raise ValueError("Existing follow-up reference changed")
        return path

    def measure(
        family,
        thread,
        role,
        stem,
        models,
        routing=None,
        repeats=1,
        block=0,
        kind="configuration",
        method="selected",
        freeze_path=None,
    ):
        refs = directory / f"references-{role}.json"
        request = {
            "registry": str(registry_paths[family].resolve()),
            "registry_sha256": plan["registry_hashes"][family],
            "workload": str((directory / f"{role}.json").resolve()),
            "workload_sha256": plan["workload_hashes"][role],
            "references": str(refs.resolve()),
            "references_sha256": sha256(refs),
            "role": role,
            "kind": kind,
            "threads": thread,
            "models": models,
            "routing": routing or {},
            "warmup": warmup,
            "repeats": repeats,
            "order_seed": seed + block,
            "block": block,
            "method": method,
            "condition_id": f"{family}-t{thread}-{method}",
        }
        if freeze_path is not None:
            request.update(
                {"freeze": str(freeze_path.resolve()), "freeze_sha256": sha256(freeze_path)}
            )
        request_path, result_path = (
            measurements / f"{stem}.request.json",
            measurements / f"{stem}.json",
        )
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("Existing follow-up measurement request changed")
        if not result_path.exists():
            write_json(request_path, request)
            print(f"Running {stem}", flush=True)
            subprocess.run(
                worker + ["followup", "--request", str(request_path), "--output", str(result_path)],
                check=True,
            )
        result = read_json(result_path)
        if result["request_sha256"] != sha256(request_path):
            raise ValueError("Cached follow-up result belongs to different inputs")
        return result

    reference("fit")
    jobs = [
        (family, thread, name)
        for family in available
        for thread in threads
        for name, graph in registries[family]["graphs"].items()
        if graph["passed"]
    ]
    random.Random(seed).shuffle(jobs)
    profiles = {family: {str(thread): {} for thread in threads} for family in available}
    fit_rows = read_json(directory / "fit.json")["rows"]
    for family, thread, name in jobs:
        if name != "dynamic" and not any(
            row["tokens"] <= int(name.split("-")[1]) for row in fit_rows
        ):
            continue
        result = measure(
            family,
            thread,
            "fit",
            f"profile-{family}-t{thread}-{name}",
            [name],
            repeats=2,
            kind="profile",
            method=name,
        )
        profiles[family][str(thread)][name] = result
    policies_path = directory / "fit-policies.json"
    configuration_jobs = []
    for family in available:
        for thread in threads:
            options = {
                name: result
                for name, result in profiles[family][str(thread)].items()
                if result["passed"]
            }
            if "dynamic" not in options:
                continue
            for group in subsets([name for name in options if name != "dynamic"]):
                routing = learn_routes(options, group)
                if all(name in routing.values() for name in group):
                    configuration_jobs.append((family, thread, group, routing))
    random.Random(seed + 1).shuffle(configuration_jobs)
    fit_results = {family: {str(thread): [] for thread in threads} for family in available}
    for family, thread, group, routing in configuration_jobs:
        suffix = "-".join(group) if group else "dynamic"
        result = measure(
            family,
            thread,
            "fit",
            f"config-{family}-t{thread}-{suffix}",
            ["dynamic", *group],
            routing=routing,
        )
        fit_results[family][str(thread)].append(result)
    policies = {}
    conditions = {}
    for family in available:
        policies[family] = {}
        for thread in threads:
            valid = [item for item in fit_results[family][str(thread)] if item["passed"]]
            try:
                chosen = select_configuration(valid, budget_mib * 2**20)
            except ValueError:
                continue
            policy = {
                "family": family,
                "threads": thread,
                "variants": chosen["variants"],
                "routing": chosen["routing"],
                "fit_p95_ms": chosen["latency"]["end_to_end_ms"]["p95_ms"],
                "fit_peak_bytes": chosen["memory_for_budget_bytes"],
                "fit_workload_sha256": plan["workload_hashes"]["fit"],
                "registry_sha256": plan["registry_hashes"][family],
            }
            policies[family][str(thread)] = policy
            for method in ("dynamic", "selected"):
                conditions[f"{family}-t{thread}-{method}"] = {
                    "family": family,
                    "threads": thread,
                    "method": method,
                    "models": ["dynamic", *policy["variants"]]
                    if method == "selected"
                    else ["dynamic"],
                    "routing": policy["routing"] if method == "selected" else {},
                    "registry_sha256": plan["registry_hashes"][family],
                }
    if policies_path.exists() and read_json(policies_path) != policies:
        raise ValueError("Fitted policies changed after being saved")
    write_json(policies_path, policies)
    confirmation_freeze = directory / "confirmation-plan.json"
    confirmation_plan = {
        "stage": "fit_policies_frozen",
        "conditions": conditions,
        "workload_hashes": plan["workload_hashes"],
        "experiment_sha256": sha256(plan_path),
        "fit_policies_sha256": sha256(policies_path),
    }
    if confirmation_freeze.exists() and read_json(confirmation_freeze) != confirmation_plan:
        raise ValueError("Confirmation plan changed")
    write_json(confirmation_freeze, confirmation_plan)
    reference("confirmation")

    def phase(role, phase_conditions, freeze_path):
        results = []
        for block in range(blocks):
            ordered = list(phase_conditions.items())
            random.Random(seed + 20 + block).shuffle(ordered)
            for identifier, condition in ordered:
                results.append(
                    measure(
                        condition["family"],
                        condition["threads"],
                        role,
                        f"{role}-b{block}-{identifier}",
                        condition["models"],
                        routing=condition["routing"],
                        block=block,
                        method=condition["method"],
                        freeze_path=freeze_path,
                    )
                )
        return results

    confirmation = phase("confirmation", conditions, confirmation_freeze)
    groups, ratios = [], {}
    for family in available:
        family_groups, family_ratios = summarize_runs(
            [run for run in confirmation if run["family"] == family]
        )
        for group in family_groups:
            group.update(
                {
                    "family": family,
                    "variants": policies[family][str(group["threads"])]["variants"]
                    if group["method"] == "selected"
                    else [],
                }
            )
        groups.extend(family_groups)
        ratios[family] = family_ratios
    chosen = pick_candidate(groups, budget_mib * 2**20)
    write_json(
        directory / "confirmation-summary.json",
        {
            "split": "train",
            "role": "confirmation",
            "groups": groups,
            "matched_block_ratios": ratios,
            "chosen": chosen,
            "all_checks_passed": all(result["passed"] for result in confirmation),
        },
    )
    audit_conditions = {
        identifier: condition
        for identifier, condition in conditions.items()
        if condition["threads"] == chosen["threads"]
    }
    audit_freeze = directory / "audit-plan.json"
    frozen = {
        "stage": "audit_choice_frozen",
        "chosen": chosen,
        "conditions": audit_conditions,
        "workload_hashes": plan["workload_hashes"],
        "experiment_sha256": sha256(plan_path),
        "confirmation_plan_sha256": sha256(confirmation_freeze),
        "confirmation_summary_sha256": sha256(directory / "confirmation-summary.json"),
    }
    if audit_freeze.exists() and read_json(audit_freeze) != frozen:
        raise ValueError("Audit choice changed after freezing")
    write_json(audit_freeze, frozen)
    reference("audit")
    audit = phase("audit", audit_conditions, audit_freeze)
    audit_groups = []
    for family in available:
        family_groups, _ = summarize_runs([run for run in audit if run["family"] == family])
        for group in family_groups:
            group.update(
                {
                    "family": family,
                    "variants": policies[family][str(group["threads"])]["variants"]
                    if group["method"] == "selected"
                    else [],
                }
            )
        audit_groups.extend(family_groups)
    summary = {
        "status": "complete",
        "split": "train",
        "purpose": "exploratory inference optimization audit",
        "chosen_before_audit": chosen,
        "audit_groups": audit_groups,
        "confirmation_groups": groups,
        "audit_plan_sha256": sha256(audit_freeze),
        "fit_policies_sha256": sha256(policies_path),
        "all_audit_checks_passed": all(result["passed"] for result in audit),
        "limitations": [
            "All new rows are from classifier training data, held out only from inference-policy tuning.",
            "No GLUE validation results were used or rerun in this follow-up.",
            "Three process blocks on one laptop; no confidence intervals or causal fusion attribution.",
            "The original deployment policy remains unchanged.",
        ],
    }
    write_json(directory / "summary.json", summary)
    return summary
