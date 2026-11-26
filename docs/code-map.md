# Code map

The repository implements experiment design, training, export, routing, benchmarking, and analysis. The pretrained DistilBERT weights and datasets are upstream assets. PyTorch, ONNX, ONNX Runtime, OpenVINO, and the plotting and analysis libraries are upstream dependencies. The [README](../README.md) gives the run order and the report entry point.

## Core package

| File | Components |
| --- | --- |
| [src/etdl/\_\_init\_\_.py](../src/etdl/__init__.py) | Initializes the package. |
| [src/etdl/\_\_main\_\_.py](../src/etdl/__main__.py) | Dispatches `python -m etdl` to the CLI. |
| [ablations.py](../src/etdl/ablations.py) | `workload_quantile` computes a request-weighted percentile. `reanalyse` replays configuration selection under alternate caps, checks, and length distributions. Its `describe` helper extracts the selected configuration metrics. |
| [benchmark.py](../src/etdl/benchmark.py) | `latency_summary` summarizes validated request timings. `sampled_memory` runs a background `observe` sampler for process RSS. `benchmark` measures the verified dynamic model. |
| [boundaries.py](../src/etdl/boundaries.py) | `bucket_capacity` maps token counts to capacities. `synthetic_cases` builds exact-length diagnostic inputs. `power_state` and `same_power` record and compare power conditions. `Status` maps the Windows power status structure. `cpu_observations` uses `observe` to sample CPU usage. `code_hashes` and `checked_plan` verify the frozen protocol. `reference` computes independent logits. `run_worker` times the three padding/shape paths with its per-request `execute` helper. `summarize` aggregates matched ratios. `execute` plans and runs the measurement blocks. |
| [checks.py](../src/etdl/checks.py) | `compare_logits` checks shape, finiteness, logit tolerance, and label agreement. `verified_dynamic` gates timing on a valid export check. `check` audits real and synthetic inputs with the nested `compare` helper. |
| [cli.py](../src/etdl/cli.py) | `positive` and `nonnegative` validate numeric CLI arguments. `parser` registers commands and options. `main` dispatches them. |
| [comparison.py](../src/etdl/comparison.py) | `schedule` randomizes process blocks. `method_models` and `choose_route` define resident graphs and the request route. `prepare_workload` selects matching text rows with split constraints. `references` generates PyTorch outputs. `validate_plan` and `validate_frozen_plan` check artifact integrity. `measure` runs a timed process using `infer` for backend dispatch and `execute` for per-request measurement. `deployment_candidate` selects the qualifying configuration. `freeze` writes the policy before validation. `evaluate` runs frozen validation. `summarize_runs` aggregates at the process level. `subprocess_worker` starts an isolated run. `compare` orchestrates the comparison. |
| [config.py](../src/etdl/config.py) | `Config` holds runtime and model settings. `load_config` validates TOML and resolves artifact paths relative to the config file. |
| [data.py](../src/etdl/data.py) | `model_source` pins the model and tokenizer revision. `load_tokenizer` loads that tokenizer. `encode` produces NumPy inputs while rejecting overlength text. `prepare` samples and records training calibration rows. `load_calibration` verifies and loads that sample. |
| [followup.py](../src/etdl/followup.py) | `source_graphs` locates the verified graph family. `partition_rows` makes disjoint splits. `prepare` freezes the workload. `verify_graph` checks equivalence using its `compare` helper. `build_family` exports the requested graph family. `load_registry` verifies graph metadata. `benchmark` measures a condition. `pick_candidate` selects from eligible repeats. `execute` runs the follow-up using `reference` to prepare expected outputs, `measure` to launch or recover one run, and `phase` to execute a split's schedule. |
| [io.py](../src/etdl/io.py) | `write_json` and `read_json` serialize experiment records. `sha256` fingerprints file bytes. |
| [model.py](../src/etdl/model.py) | `load_pytorch` loads the pinned teacher. `export` builds its ONNX graph with a `Logits` wrapper whose `__init__` stores the classifier and whose `forward` returns logits. `checked_model_path` resolves the expected export. |
| [router.py](../src/etdl/router.py) | `RoutedClassifier.__init__` loads the allowed ONNX sessions. `RoutedClassifier.predict` tokenizes, selects a session with a dynamic fallback, and returns the prediction and optional timing record. |
| [runtime.py](../src/etdl/runtime.py) | `environment` records machine and software information. `session` constructs a configured CPU ONNX Runtime session. `graph_operators` inventories graph operator types. |
| [selection.py](../src/etdl/selection.py) | `subsets` enumerates allowed resident variants. `route_name` resolves the trained route or dynamic fallback. `learn_routes` chooses the fastest verified route by observed length. `select_configuration` minimizes measured p95 subject to output and memory constraints. |
| [study.py](../src/etdl/study.py) | `hardware_key` identifies the relevant machine configuration. `run_worker` performs a timed fixed-shape study process through its per-request `execute` helper. `study` orchestrates calibration and policy selection through its `measure` helper. `write_study_outputs` emits the study's derived artifacts. |
| [variants.py](../src/etdl/variants.py) | `pad_inputs` extends masks and IDs without truncation. `weights_digest` fingerprints initializers. `build_variant` specializes an input shape. `check_variant` uses `compare` to validate logits on padded and boundary inputs. `verified_variants` loads variants that passed the checks. |
| [worker.py](../src/etdl/worker.py) | `main` dispatches an isolated subprocess to the requested experiment worker. |

## Training, export, and evaluation scripts

| File | Components |
| --- | --- |
| [training_feasibility.py](../scripts/training_feasibility.py) | `main` checks whether CPU training steps run within practical time and memory. |
| [train_students.py](../scripts/train_students.py) | `arguments` reads experiment options. `batches` and `model_batch` construct minibatches and tensors. `student_from_weights` copies the teacher's embeddings, initial layers, and classifier. `evaluate` scores development predictions. `main` creates partitions, trains, and writes checkpoints. |
| [train_capacity.py](../scripts/train_capacity.py) | `arguments` reads the output and training settings. `main` applies the development-gap rule and trains the four-layer student. |
| [train_cost_weighted.py](../scripts/train_cost_weighted.py) | `arguments` reads the training and output settings. `main` fits the four-layer student with length-dependent distillation weights. |
| [evaluate_students.py](../scripts/evaluate_students.py) | `arguments` parses input and output paths. `predict` produces development or validation logits. `student_model` reconstructs a student checkpoint. `truncated_teacher` creates the unchanged-weight control. `main` audits the frozen models on validation. |
| [export_students.py](../scripts/export_students.py) | The `Logits` wrapper's `__init__` stores the model and `forward` returns its logits for export. `arguments` reads the study paths. `main` exports student models and checks ONNX agreement. |
| [benchmark_students.py](../scripts/benchmark_students.py) | `arguments` reads timing options. `percentile` computes a timing quantile. `worker` measures one model in a fresh process. `parent` prepares and summarizes process blocks. |
| [plot_student_study.py](../scripts/plot_student_study.py) | `arguments` reads source and output paths. `point` plots one operating point with its measured range. `main` draws student and cascade tradeoffs. |
| [evaluate_cost_extension.py](../scripts/evaluate_cost_extension.py) | `arguments` reads input paths. `student_from_checkpoint` loads the uniform or weighted student. `predict` scores text. `paired_interval` resamples paired accuracy differences. `scores` summarizes accuracy by group. `comparison` generates paired outcomes. `main` evaluates the weighted checkpoint. |

## Gate fitting, ablations, and timing

| File | Components |
| --- | --- |
| [fit_gate.py](../scripts/fit_gate.py) | `arguments` reads experiment options. `length_bin` assigns cost groups. `model_logits` collects logits and teacher timing. `base_features` computes margin and log length. `design` adds interactions and intercepts. `score` applies coefficients and cost. `route` applies the fixed cutoff. `main` fits and freezes original policies. |
| [evaluate_gate.py](../scripts/evaluate_gate.py) | `arguments` reads the frozen gate and evaluation inputs. `paired_bootstrap_ci` resamples aligned predictions. `main` replays the original policy curves on validation. |
| [benchmark_cascade.py](../scripts/benchmark_cascade.py) | `arguments` reads the process design. `percentile` summarizes latency. `require_ac` checks power. `method_gate` resolves the selected score and cutoff. `worker` measures one policy with both models resident. `parent` prepares identical inputs, schedules processes, and aggregates results. |
| [soft_cost_gate.py](../scripts/soft_cost_gate.py) | `arrays` aligns correctness flags and teacher cost. `calibrate` selects an exponent and cutoff. `evaluate` reports fixed-cutoff and equal-call outcomes. `external_rows` selects and scores IMDb with `fingerprint` normalizing text for deduplication. `main` dispatches calibration, evaluation, and timing-plan preparation. |
| [search_gate_formulas.py](../scripts/search_gate_formulas.py) | `features` builds the design matrix. `formulas` enumerates score variants. `formula_score` applies a candidate. `top_mask` chooses the highest scores at a fixed quota. `freeze` chooses formulas without the Stanford test scores. `normalized` normalizes text for overlap checks. `predict_test` scores eligible Stanford trees. `evaluate_test` reports fixed and matched-call comparisons. `main` dispatches freezing and evaluation. |
| [sweep_cost_exponent.py](../scripts/sweep_cost_exponent.py) | `load_rows` verifies archived row hashes. `breakpoints` derives exponents where rankings can swap. `metrics` summarizes predictions, calls, and cost. `scan` evaluates the intervals. `main` freezes the development policies and tests transfer. |
| [study_gate_signals.py](../scripts/study_gate_signals.py) | `signals`, `margins`, and `feature_matrix` build lexical, logit, and standardized features. `sigmoid` maps logits to probabilities. `logistic` fits regularized models using `loss`. `group_index` assigns confidence and length groups. `fit_scores` fits candidate models and `raw_score` applies them. `percentile_rank`, `blend_score`, and `order` convert or combine ranking signals. `get_older_cohorts` loads archived inputs. `patterns` summarizes where the teacher fixes or harms. `cohort` assembles arrays for evaluation. `route_metrics` and `mask_ranked` score decisions at a given quota. `freeze` uses `preference` to rank finalists. `infer_imdb` locks unused review IDs and computes logits. `evaluate` audits the frozen policies. `main` dispatches the two stages. |
| [study_teacher_harm.py](../scripts/study_teacher_harm.py) | `quality` predicts harm or fix-minus-harm. `adjust_score` reorders requests near the cutoff. `read_reference_imdb` verifies the earlier IMDb archive. `freeze` uses `preference` to select finalists. `evaluate` measures their matched and fixed-cutoff outcomes. `main` dispatches calibration and fresh evaluation. |
| [plot_gate_grid.py](../scripts/plot_gate_grid.py) | `evaluate_grid` scores all exponent and cutoff combinations. `selected_metrics` audits a fixed policy. `grid_point` serializes a cell. `transfer_shortlist` picks the retrospective two-split policy. `check_imdb` freezes and tests both choices on unused reviews. `plot_grid` draws the heatmaps. `main` runs the grid or IMDb stage. |
| [time_gate_grid.py](../scripts/time_gate_grid.py) | `main` verifies the frozen grid policy and starts the matched CPU run. `summarize` audits live predictions and records paired latency and memory. |
| [run_int8_extension.py](../scripts/run_int8_extension.py) | `arguments` reads quantization and benchmark settings. `require_ac` checks power. `source_paths`, `validate_sources`, and `candidate_path` resolve inputs and verify their hashes. `strip_shape_metadata` prepares the exploratory repair. `make_candidate` quantizes and checks the graph. `confidence_fallback` applies the frozen cutoff. `audit_validation` and `audit_external` collect prediction evidence. `method_student` maps policies to model precision. `expected_for_method` returns expected routing and prediction. `benchmark_worker` measures a policy. `summarize` uses `paired_reduction` to compare blocks. `main` coordinates the audit, retry, and timing stages. |
| [recalibrate_int8.py](../scripts/recalibrate_int8.py) | `confidence_scores` computes the INT8 gate score. `audit_rows` counts changed outcomes. `read_old` verifies original exports by hash. `calibrate` fits new cutoffs on Stanford. `replay` applies them to GLUE and IMDb. `main` dispatches calibration and replay. |

## Tools and report builders

| File | Components |
| --- | --- |
| [archive_study.py](../tools/archive_study.py) | `main` copies the initial fixed-shape evidence to the tracked archive. |
| [archive_comparison.py](../tools/archive_comparison.py) | `csv_rows` builds tabular exports. `capture_traces` records representative inference traces. `figures` renders comparisons. `main` assembles the archive. |
| [archive_followup.py](../tools/archive_followup.py) | `label` formats condition names. `render` builds plots. `diagnostics` summarizes auxiliary measurements. `main` packages the follow-up results. |
| [archive_boundaries.py](../tools/archive_boundaries.py) | `telemetry_gap` checks coverage of CPU sampling. `render` draws the boundary plots. `main` archives the diagnostic. |
| [breakeven_map.py](../tools/breakeven_map.py) | `config` loads settings. `code_hashes` fingerprints the measurement implementation. `graph_paths` resolves verified models. `exclusions` lists previously used samples. `prepare` freezes exact-length rows. `checked` verifies the experiment inputs. `references` computes expected logits. `openvino_sessions` compiles the reference and fixed paths. `worker` times one process using `infer` for backend-specific inference and `execute` for each request. `schedule` orders the blocks. `run` launches them. `summarize` assembles the break-even measurements. |
| [real31.py](../tools/real31.py) | `config`, `code_hashes`, and `graph_paths` resolve and verify supporting inputs. `prepare` freezes the 31-token workload. `checked` audits it and `references` builds expected predictions. `openvino_sessions` compiles both paths. `worker` measures one process with `infer` and `execute` helpers for model invocation and per-request timing. `run` orchestrates the study. `summarize` aggregates the paired results. |
| [plot_breakeven_map.py](../tools/plot_breakeven_map.py) | `errorbar` draws a median and process range. `main` renders the fixed-shape break-even analysis. |
| [build_project_report.py](../tools/build_project_report.py) | `verify_archive` checks historical evidence. `render_tradeoff` draws its tradeoff plot. `main` builds the original fixed-shape report with `paragraph` and `literal` helpers for prose and code blocks. |
| [demo.py](../tools/demo.py) | `main` demonstrates the original frozen fixed-shape study policy with reusable sessions. |
| [plot_report_routing.py](../tools/plot_report_routing.py) | `load` reads archived result files. `draw_grid` produces the report's accuracy heatmap. `draw_latency` plots paired median and tail changes. |
| [report/build.ps1](../report/build.ps1) | Compiles the NeurIPS report with the official style and writes the submission PDF. |
| [presentation/build_deck.ps1](../presentation/build_deck.ps1) | Rebuilds the archived NC State presentation in PowerPoint. |

## Tests

| File | What it checks |
| --- | --- |
| [test_ablations.py](../tests/test_ablations.py) | Correct workload weighting and rejection of invalid latency records. |
| [test_boundaries.py](../tests/test_boundaries.py) | Boundary lengths, padding semantics, protocol hashes, pairing, and power state. |
| [test_boundary_archive.py](../tests/test_boundary_archive.py) | Detection of missing and interrupted telemetry. |
| [test_comparison.py](../tests/test_comparison.py) | Reproducible schedules, safe fallbacks, sample disjointness, frozen validation, and process-level analysis. |
| [test_contracts.py](../tests/test_contracts.py) | Config paths, tokenizer pinning, split isolation, input length, numerical checks, and benchmark invariants. |
| [test_followup.py](../tests/test_followup.py) | Disjoint partitions, reserved audits, frozen conditions, and selection constraints. |
| [test_router.py](../tests/test_router.py) | Safe dynamic fallback and omission of raw request text from route logs. |
| [test_selection.py](../tests/test_selection.py) | Route decisions, loaded variants, measured memory, and p95 selection. |
| [test_variants.py](../tests/test_variants.py) | Padding, overflow, weight preservation, and execution of specialized shapes. |
