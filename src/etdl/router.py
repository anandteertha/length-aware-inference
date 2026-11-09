"""Load a frozen, checked policy with an always-available dynamic fallback."""

import json
import time
from datetime import UTC, datetime

import numpy as np

from etdl.checks import verified_dynamic
from etdl.data import encode, load_tokenizer, model_source
from etdl.io import read_json, sha256
from etdl.runtime import environment, session
from etdl.selection import route_name
from etdl.study import hardware_key
from etdl.variants import pad_inputs, verified_variants


class RoutedClassifier:
    def __init__(self, config, policy_path):
        started = time.perf_counter_ns()
        self.config = config
        self.policy = read_json(policy_path)
        policy = self.policy
        if policy.get("status") != "verified_calibration_selection" or len(policy["variants"]) > 2:
            raise ValueError("Invalid or unverified selection policy")
        if policy["model_source"] != model_source(config):
            raise ValueError("Selection belongs to another model source")
        if policy["calibration_sha256"] != sha256(config.artifact_dir / "data/calibration.json"):
            raise ValueError("Selection belongs to different calibration data")
        if (policy["atol"], policy["rtol"]) != (config.atol, config.rtol):
            raise ValueError("Selection output tolerances differ")
        if policy["hardware_key"] != hardware_key(environment()):
            raise ValueError("Selection was measured on different hardware/runtime; rerun study")
        if policy["measured_memory_bytes"] > policy["memory_budget_bytes"]:
            raise ValueError("Selection exceeds its measured memory budget")
        paths = {"dynamic": verified_dynamic(config)}
        if sha256(paths["dynamic"]) != policy["model_hashes"].get("dynamic"):
            raise ValueError("Dynamic fallback changed after selection")
        self.disabled_variants = {}
        for name in policy["variants"]:
            try:
                candidate = verified_variants(config, [], names=[name]).get(name)
                if candidate is None or sha256(candidate) != policy["model_hashes"].get(name):
                    raise ValueError("Selected variant is unavailable or changed")
                paths[name] = candidate
            except (OSError, ValueError, KeyError) as error:
                self.disabled_variants[name] = str(error)
        self.tokenizer = load_tokenizer(config)
        settings = policy.get("runtime_settings", {})
        options = (
            {"allow_spinning": settings["allow_spinning"]} if "allow_spinning" in settings else {}
        )
        self.runtimes = {
            name: session(path, policy["threads"], **options)[0] for name, path in paths.items()
        }
        self.ready_ms = (time.perf_counter_ns() - started) / 1e6

    def predict(self, text):
        started = time.perf_counter_ns()
        inputs = encode(self.tokenizer, text, self.config.max_length)
        length = inputs["input_ids"].shape[1]
        name = route_name(length, self.policy["routing"], self.runtimes)
        if name != "dynamic":
            inputs = pad_inputs(inputs, int(name.split("-")[1]), self.tokenizer.pad_token_id)
        logits = self.runtimes[name].run(["logits"], inputs)[0]
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite output")
        prediction = int(logits.argmax(-1)[0])
        result = {
            "prediction": prediction,
            "label": "positive" if prediction else "negative",
            "logits": logits[0].tolist(),
            "tokens": length,
            "route": name,
            "disabled_variants": self.disabled_variants,
            "end_to_end_ms": (time.perf_counter_ns() - started) / 1e6,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        # Log the decision without storing the user's input text.
        path = self.config.artifact_dir / "reports/routing-decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        return result
