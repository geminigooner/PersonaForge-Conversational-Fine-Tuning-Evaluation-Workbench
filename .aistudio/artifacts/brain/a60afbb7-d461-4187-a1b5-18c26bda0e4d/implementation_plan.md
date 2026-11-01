# Gemini Corpus & Tinman (`Tinman-Lab/Tinman-gemma4-companion-merged`) LoRA Fine-Tuning & Evaluation Workbench

A local-first SQLite fine-tuning and behavioral evaluation workbench for curating, separating, and comparing multi-turn outputs across **Gemini 2.5 Pro (`google/gemini-2.5-pro`), Gemini 3.1 Pro (`gemini-3.1-pro-preview`), and Gemini 3 Flash (`gemini-3-flash`)** (as well as `gemini-3.8-flash`, `gemini-4-argon`, and custom Gemini checkpoints), compiling model-segregated Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) datasets with 4-layer leakage protection, verifying and configuring **LoRA/QLoRA adapters on top of `Tinman-Lab/Tinman-gemma4-companion-merged`**, tracking 5-stage **Parent Checkpoint Lineage**, and running a reproducible **Model Evaluation & Regression Harness**—styled in the dreamy **Violet, Lavender, Warm-Cream, Deep-Plum & Bubblegum-Pink aesthetic** from your reference images.

---

## Pre-Implementation Verification Gate & Core Architecture

> [!WARNING]
> **Mandatory Pre-Flight Verification Rule**: The workbench does **not** assume unverified chat template names, target module lists, or model classes prior to inspecting the target checkpoint. Training script generation is gated behind a **Checkpoint & Environment Introspection Manifest** verified against `Tinman-Lab/Tinman-gemma4-companion-merged` and the active local Python environment.

1. **Multi-Model Gemini Teacher Separation (`Gemini 2.5 Pro`, `Gemini 3.1 Pro`, `Gemini 3 Flash`)**:
   - **First-Class Model Presets & Attribution**: Explicitly includes **Gemini 2.5 Pro** (`google/gemini-2.5-pro`), **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`), and **Gemini 3 Flash** (`gemini-3-flash`) as dedicated primary teacher models (alongside live SDK models and custom strings) with distinct color-coded **Model Attribution Badges** on every turn row, conversation message bubble, and DPO card.
   - **1-Click Model Separation & Grouped Partition Views**: Users can click top-level partition pills (`All Models` | `Gemini 2.5 Pro` | `Gemini 3.1 Pro` | `Gemini 3 Flash` | `+ Custom`) to isolate outputs from a single model, or toggle **"Group by Source Model"** to view dedicated per-model buckets comparing turn counts, eligibility rates, and average `Narrative Continuity`, `Character Consistency`, and `Emotional Expressiveness` scores across **2.5 Pro**, **3.1 Pro**, and **3 Flash**.
   - **Model-Segregated SFT/DPO Dataset Compilation**: Filter SFT or DPO `.jsonl` exports to a single teacher model (e.g., *only* `Gemini 2.5 Pro` or *only* `Gemini 3.1 Pro`), pair one model as `Chosen` against another as `Rejected` (e.g., `2.5 Pro` or `3.1 Pro` vs. `3 Flash` in DPO), and record the exact `teacher_model_breakdown_json` inside every `dataset_version` in the 5-Stage Lineage.
2. **Parent Training Target vs. Root Baseline Separation**:
   - **Active Parent Model for LoRA Training**: `Tinman-Lab/Tinman-gemma4-companion-merged` (full-precision merged companion checkpoint). All new SFT and DPO LoRA adapters attach directly to `Tinman-Lab/Tinman-gemma4-companion-merged` as their parent base model.
   - **Preserved Root & Intermediate Baselines**: `google/gemma-4-E4B-it` (untouched root model) and the intermediate **Abliterated Gemma 4 E4B** checkpoint are preserved as separate parent lineage nodes and baseline comparison targets in the Evaluation Harness.
3. **Corrected Context Window (128K) & Checkpoint-Native Tokenizer/Template Verification**:
   - **Documented Context Window**: Corrected to **128K tokens (`131,072`)** for the Gemma 4 E4B architecture.
   - **Tokenizer & Chat Template Introspection**: Rather than assuming `"gemma-4-thinking"` is a registered template identifier in Unsloth or Transformers, the workbench inspects the actual `tokenizer_config.json` and `chat_template` Jinja string bundled inside `Tinman-Lab/Tinman-gemma4-companion-merged`.
   - **Verified Control Tokens**: Uses the checkpoint's native `tokenizer.apply_chat_template(...)` (or `processor.apply_chat_template(...)`) and verified `<bos>`, `<|turn>system`, `<|turn>user`, `<|turn>model`, `<turn|>`, and documented `<|think|>` control mechanisms only as defined by the inspected checkpoint template and installed `transformers` version.
4. **Experimental Hyperparameter Controls & Dynamic Target-Module Introspection**:
   - **Configurable Optimization & Monitoring**: Proposed PLE learning rates (`2e-4` for SFT, `5e-6` for DPO) are treated strictly as **experimental starting points**, not rigid architectural rules. Users have full interactive control over:
     - `learning_rate` & `lr_scheduler_type` (`cosine`, `linear`, `constant_with_warmup`)
     - `warmup_ratio` and `warmup_steps`
     - `max_grad_norm` (gradient clipping threshold, e.g., `1.0` or `0.5`)
     - `weight_decay` and optimizer choice (`adamw_8bit`, `paged_adamw_32bit`, `adamw_torch`)
     - Validation split ratio (`val_split_ratio`), `eval_strategy="steps"`, `eval_steps`, `save_strategy`, and validation-loss monitoring (`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`).
   - **Runtime Target-Module Verification**: Before training, the generated introspection probe enumerates `[name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]` on the loaded `Tinman-Lab/Tinman-gemma4-companion-merged` checkpoint to confirm exact projection layer paths (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and whether multimodal tower layers are present) before instantiating `LoraConfig`.
5. **Five-Stage Recursive Parent-Checkpoint Lineage**:
   - The SQLite schema models explicit parent-child derivation across all five evolutionary stages:
     1. **Stage 0 — Root Baseline**: `google/gemma-4-E4B-it` (Original Google Gemma 4 E4B IT, 128K context window).
     2. **Stage 1 — Abliterated Checkpoint**: Abliterated Gemma 4 E4B checkpoint (`parent_checkpoint_id` $\rightarrow$ `google/gemma-4-E4B-it`).
     3. **Stage 2 — Tinman SFT/DPO & Merged Checkpoint**: `Tinman-Lab/Tinman-gemma4-companion-merged` (`parent_checkpoint_id` $\rightarrow$ Abliterated Gemma checkpoint, incorporating Tinman’s SFT/DPO training and merge).
     4. **Stage 3 — Amanda’s Curated Datasets & LoRA Experiments**: Leak-verified `dataset_versions` (recording exact `Gemini 2.5 Pro`, `Gemini 3.1 Pro`, and `Gemini 3 Flash` teacher breakdowns) and `lora_configs` / `training_runs` targeting `Tinman-Lab/Tinman-gemma4-companion-merged`.
     5. **Stage 4 — Resulting Adapter Checkpoints & Evaluation Runs**: Produced LoRA `adapter_checkpoints` and comparative `eval_runs` benchmarked against Stage 0, Stage 1, and Stage 2 baselines.

---

## Implementation Status Matrix (Implemented vs. Proposed Features)

| Module / Capability | Current Status | Scope & Implementation Details |
| :--- | :--- | :--- |
| **React 19 + Vite + Tailwind v4 Shell** | **Implemented (Baseline)** | Base application scaffold (`src/App.tsx`, `vite.config.ts`) and `@google/genai` SDK installed. |
| **SQLite Engine & 5-Stage Recursive Lineage DAG** | **Proposed (Core Build)** | Self-referencing `model_checkpoints` table preserving Stage 0 (`google/gemma-4-E4B-it`) $\rightarrow$ Stage 1 (Abliterated) $\rightarrow$ Stage 2 (`Tinman-Lab/Tinman-gemma4-companion-merged`) $\rightarrow$ Stage 3 (Amanda's Datasets & LoRA Runs) $\rightarrow$ Stage 4 (Adapters & Evals). |
| **Gemini 2.5 Pro, 3.1 Pro & 3 Flash Separation** | **Proposed (Core Build)** | Dedicated filter pills, grouped buckets, attribution badges, per-model rubric scorecards, and segregated JSON/JSONL exports for `Gemini 2.5 Pro`, `Gemini 3.1 Pro`, and `Gemini 3 Flash`. |
| **Pre-Flight Checkpoint & Environment Verifier** | **Proposed (Core Build)** | Generates `verify_tinman_checkpoint.py` and ingests/validates the JSON introspection report (`config.json`, `tokenizer_config.json`, linear modules, library versions) before generating training scripts. |
| **Training Corpus & Multi-Turn Thread Logger** | **Proposed (Core Build)** | Manual paste (`Gemini 2.5 Pro`, `Gemini 3.1 Pro`, `Gemini 3 Flash`, etc.) + Teacher generation, immutable `original_transcript` preservation, date/model filtering, and 1–10 rubric scoring. |
| **4-Layer Data Leakage Firewall & Audit Log** | **Proposed (Core Build)** | Normalized SHA-256 exact matching, n-gram Jaccard near-duplicate detection, multi-turn conversation subsequence checks, and an immutable `leakage_audit_log` for manual overrides. |
| **Tinman SFT & DPO Dataset Studio** | **Proposed (Core Build)** | Formats multi-turn SFT and Chosen/Rejected DPO pairs (including cross-Gemini-model Chosen vs. Rejected comparisons) using verified `Tinman-Lab/Tinman-gemma4-companion-merged` tokenizer rules. |
| **Configurable LoRA Architect & Validation Monitor** | **Proposed (Core Build)** | Targets `Tinman-Lab/Tinman-gemma4-companion-merged` (128K context) with configurable LR, warmup, gradient clipping (`max_grad_norm`), validation-loss monitoring, and verified target modules. |
| **Local Checkpoint Inference Bridge + Teacher/Judge Separation** | **Proposed (Core Build)** | Connects to local OpenAI-compatible endpoints (vLLM, Ollama, `llama.cpp`) or imports local batch JSONL outputs; isolates cloud Gemini API strictly to Teacher/Judge roles. |
| **Independent Human + LLM-as-a-Judge & Inter-Rater Agreement** | **Proposed (Core Build)** | Preserves Human and Judge scores separately, computes Inter-Rater Agreement (MAD, rank correlation, bias delta), and requires concise evidence-quoted rationales. |
| **Reproducibility & Multi-Baseline Regression Dashboard** | **Proposed (Core Build)** | Compares new LoRA adapters against `Tinman-Lab/Tinman-gemma4-companion-merged`, Abliterated Gemma, and `google/gemma-4-E4B-it` across all 6 behavioral dimensions. |

---

## 1. Visual Identity & Aesthetic Direction (From Reference Images)

- **Aesthetic Direction**: Tactile, kawaii-inspired **Purple, Lavender, Warm-Cream, Deep-Plum & Bubblegum-Pink Fine-Tuning Workbench** (strictly a fine-tuning and behavioral evaluation workbench—no RAG, no fake OS desktop toys, and no fabricated hardware stats).
- **Color Palette & Tokens**:
  - **Deep Violet & Twilight Purple (`#231145` $\rightarrow$ `#5E2CA5`)**: Primary outer backdrop, window headers, and navigation chrome.
  - **Warm Cream (`#FFF8EC`)**: Readable main laboratory canvas, long-form transcript editors, and side-by-side comparison wells.
  - **Lavender & Soft Lilac (`#D8B4FE` / `#F3E8FF`)**: Secondary cards, lineage stage containers, per-model partition buckets, and hyperparameter panels.
  - **Bubblegum Pink (`#FF5CA8` / `#FF85C0`)**: Active navigation pill highlights, primary CTA pill buttons, and interactive accents.
  - **Deep Plum (`#2A114B`)**: Primary typography, chunky `2.5px–3px` solid card outlines, and tactile offset shadows (`0 4px 0 #2A114B`).
  - **Soft Gold (`#FACC15`) & Semantic Mint/Coral**: Verification badges, baseline markers, clean leakage indicators (`#6EE7B7`), and regression/leakage warnings (`#FDA4AF`).
- **Typography**:
  - **Headings & Pill Controls**: `Fredoka` / `Quicksand` (Bold 600–700) in `#2A114B`.
  - **Long-Form Transcripts & Judge Critiques**: `Nunito` / `Plus Jakarta Sans` (15px, `1.65` line-height) on `#FFF8EC` warm cream.
  - **Hashes, Checkpoints, Model Versions, Learning Rates & Tabular Scores**: `JetBrains Mono` (`tabular-nums`).

---

## 2. User Experience & Core Fine-Tuning Flows

1. **Multi-Model Gemini Corpus Explorer & 4-Layer Leakage Inspector**:
   - **Separate & Compare Across Gemini 2.5 Pro, 3.1 Pro, and 3 Flash**:
     - Switch between **1-Click Gemini Model Partition Pills** (`All Models`, `Gemini 2.5 Pro`, `Gemini 3.1 Pro`, `Gemini 3 Flash`, plus live SDK variants and custom model strings) or toggle **"Group by Source Model"** to view dedicated per-model buckets.
     - View **Per-Model Teacher Analytics** showing exact turn counts, training-eligible percentage, and mean `Narrative Continuity`, `Character Consistency`, and `Emotional Expressiveness` scores for **Gemini 2.5 Pro**, **Gemini 3.1 Pro**, and **Gemini 3 Flash** side-by-side.
     - Every turn row and conversation bubble carries a color-coded **Model Attribution Badge** (`model_version` + `provider_source` + `recorded_at` date).
   - **Immutable Transcript Preservation & Leakage Guard**:
     - Compare immutable `original_transcript` vs. curated `model_output`.
     - Real-time **4-Layer Leakage Firewall** checks every turn and full conversation context against `eval_benchmarks` (Exact Normalized SHA-256, N-Gram Near-Duplicate Similarity, Conversation Subsequence scan, and Auditable Manual Overrides logged in `leakage_audit_log`).
2. **Guided SFT & DPO Pair Studio (With Model-Segregated Exports)**:
   - Construct multi-turn SFT records and side-by-side `Chosen` vs. `Rejected` DPO preference pairs.
   - **Cross-Model DPO Pairing**: Easily pair a high-scoring output from `Gemini 2.5 Pro` or `Gemini 3.1 Pro` (as `Chosen`) against a flatter response from `Gemini 3 Flash` (as `Rejected`), recording both `chosen_model_version` and `rejected_model_version`.
   - **Model-Filtered JSON/JSONL Export (`v1.2.0`)**: Export the entire corpus, filter to a single Gemini model (e.g., *only* `Gemini 2.5 Pro`, *only* `Gemini 3.1 Pro`, or *only* `Gemini 3 Flash`), or download a partitioned JSONL bundle where every record preserves `model_version`, `provider_source`, `recorded_at`, conversation ordering, and behavioral scores.
3. **Lineage Registry & Pre-Flight Checkpoint Verifier**:
   - View the interactive **5-Stage Checkpoint Lineage Graph**:
     - **Stage 0**: `google/gemma-4-E4B-it` (Root Baseline, 128K context)
     - **Stage 1**: `Abliterated Gemma 4 E4B` (Refusal-orthogonalized intermediate)
     - **Stage 2**: `Tinman-Lab/Tinman-gemma4-companion-merged` (Tinman SFT/DPO + Merged Parent Target)
     - **Stage 3**: Amanda’s Datasets (`dataset_versions` with `2.5 Pro` / `3.1 Pro` / `3 Flash` breakdown) & LoRA Experiments (`training_runs`)
     - **Stage 4**: Resulting LoRA Adapter Checkpoints (`adapter_checkpoints`) & Evaluation Runs (`eval_runs`)
   - **Pre-Flight Introspection Workflow**: Download or run `verify_tinman_checkpoint.py` against `Tinman-Lab/Tinman-gemma4-companion-merged` (or upload its output `checkpoint_manifest.json`) to verify `architectures`, `max_position_embeddings` (`131072`), exact `torch.nn.Linear` module names, `tokenizer.chat_template`, `<|think|>` support, and installed `transformers` / `peft` / `trl` / `unsloth` versions.
4. **Configurable LoRA Architect & Validation-Loss Script Builder**:
   - Parent model defaults to **`Tinman-Lab/Tinman-gemma4-companion-merged`** (with 128K context window ceiling and verified linear target modules).
   - **Full Optimization & Regularization Controls** (treating PLE learning rates as adjustable experimental starting points):
     - Learning Rate (`learning_rate`, e.g., `1e-4`, `2e-4`, `5e-5` for SFT; `5e-6`, `1e-6` for DPO)
     - Warmup (`warmup_ratio`, `warmup_steps`) & Scheduler (`cosine`, `linear`, `constant_with_warmup`)
     - Gradient Clipping (`max_grad_norm`, default `1.0`) & Weight Decay (`weight_decay`, default `0.01`)
     - Validation Monitoring (`val_split_ratio`, `eval_strategy="steps"`, `eval_steps`, `early_stopping_patience`, `load_best_model_at_end=True`)
     - LoRA Rank (`r`), Alpha (`lora_alpha`), Dropout (`lora_dropout`), and verified Target Modules selector.
   - Generates a verification-guarded training script that asserts checkpoint class, tokenizer template, and module existence before executing `SFTTrainer` or `DPOTrainer`.
5. **Multi-Baseline Evaluation Harness & Regression Dashboard**:
   - Run or import benchmark outputs across **6 behavioral dimensions** (`persona_consistency`, `narrative_continuity`, `emotional_expressiveness`, `improvisation`, `conversational_adaptability`, `instruction_following`).
   - Compare any **Stage 4 LoRA Adapter Checkpoint** side-by-side against **Stage 2 (`Tinman-Lab/Tinman-gemma4-companion-merged`)**, **Stage 1 (Abliterated Gemma)**, **Stage 0 (`google/gemma-4-E4B-it`)**, or teacher Gemini reference baselines (`Gemini 2.5 Pro`, `Gemini 3.1 Pro`, `Gemini 3 Flash`).
   - **Separated Human & LLM-as-a-Judge Scoring**: View independent human scores, concise evidence-quoted judge scores, and **Inter-Rater Agreement (MAD, rank correlation, directional bias, divergence alerts)** alongside per-dimension regression warnings.

---

## 3. Technical Architecture & SQLite Lineage Schema

### Architecture & 5-Stage Lineage Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                   5-STAGE RECURSIVE CHECKPOINT LINEAGE (SQLite DAG)                  │
│                                                                                      │
│  [Stage 0: Root Baseline]        google/gemma-4-E4B-it (128K Context Window)         │
│               │                                                                      │
│               ▼                                                                      │
│  [Stage 1: Abliterated Base]     Gemma-4-E4B Abliterated Checkpoint                  │
│               │                                                                      │
│               ▼                                                                      │
│  [Stage 2: Tinman Parent Target] Tinman-Lab/Tinman-gemma4-companion-merged           │
│               │                  (Tinman SFT/DPO + Full-Precision Merged Parent)     │
│               ├─────────────────────────────────────────────┐                        │
│               ▼                                             ▼                        │
│  [Stage 3: Amanda's Experiments] Dataset Snapshot (v1.x) + LoRA Config + Train Run   │
│               │                  (Tracks 2.5 Pro / 3.1 Pro / 3 Flash Breakdown)      │
│               ▼                                             │                        │
│  [Stage 4: Resulting Artifacts]  LoRA Adapter Checkpoint ◄──┘                        │
│                                       │                                              │
│                                       ▼                                              │
│                                  Evaluation Run (vs. Stage 2 Tinman Merged & Stage 0)│
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Relational SQLite Schema (`better-sqlite3` WAL)

1. **`model_checkpoints` (Recursive Parent-Child Lineage Table)**:
   - `id` (TEXT PRIMARY KEY)
   - `parent_checkpoint_id` (TEXT, Nullable Foreign Key $\rightarrow$ `model_checkpoints.id`) — Explicitly links Stage 0 $\rightarrow$ Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 4 checkpoints.
   - `stage` (TEXT) — `'stage_0_root'` | `'stage_1_abliterated'` | `'stage_2_tinman_merged'` | `'stage_4_lora_adapter'`.
   - `name` (TEXT) & `hf_repo_or_path` (TEXT) — e.g., `'google/gemma-4-E4B-it'`, `'Tinman-Lab/Tinman-gemma4-companion-merged'`.
   - `revision_commit_sha` (TEXT) & `weights_sha256` (TEXT).
   - `context_window` (INTEGER, default `131072` for 128K).
   - `verified_manifest_json` (TEXT) — Stores the verified output from inspecting the actual checkpoint: `{ verified: boolean, model_class, architectures, discovered_linear_modules, chat_template_jinja, special_tokens, think_token_supported, transformers_version, trl_version, peft_version, unsloth_version, verified_at }`.
   - `provenance_notes` (TEXT).
2. **`dataset_versions` & `lora_configs` & `training_runs` (Stage 3 — Amanda's Experiments)**:
   - `dataset_versions`: `id`, `version_tag`, `split_type` (`'sft'` | `'dpo'`), `teacher_models_filter_json`, `teacher_model_breakdown_json` (exact count of turns per source Gemini model: `Gemini 2.5 Pro`, `Gemini 3.1 Pro`, `Gemini 3 Flash`, etc.), `train_count`, `val_count`, `dataset_sha256`, `leakage_status`, `override_count`, `created_at`.
   - `lora_configs`:
     - `id`, `name`, `parent_checkpoint_id` (defaults to `Tinman-Lab/Tinman-gemma4-companion-merged`), `lora_r`, `lora_alpha`, `lora_dropout`, `verified_target_modules_json`, `freeze_multimodal_towers` (`0` or `1`).
     - Configurable Optimization & Validation Fields: `learning_rate` (experimental starting point, user-editable), `lr_scheduler_type`, `warmup_ratio`, `warmup_steps`, `max_grad_norm` (gradient clipping), `weight_decay`, `optimizer_type`, `val_split_ratio`, `eval_steps`, `early_stopping_patience`, `max_seq_length` ($\le 131072$), `quantization_mode`.
   - `training_runs`: `id`, `parent_checkpoint_id`, `dataset_version_id`, `lora_config_id`, `resulting_adapter_checkpoint_id`, `train_loss_curve_json`, `val_loss_curve_json`, `best_val_loss`, `environment_versions_json`, `started_at`, `completed_at`.
3. **Training Corpus & Multi-Model Attribution (`conversations`, `outputs`, `leakage_audit_log`)**:
   - `outputs`: Stores multi-turn transcripts, `model_version` (indexed for fast per-model separation: `'google/gemini-2.5-pro'`, `'gemini-3.1-pro-preview'`, `'gemini-3-flash'`, `'gemini-3.8-flash'`, `'gemini-4-argon'`), `provider_source` (`'OpenRouter'`, `'AI Studio'`, `'Live API'`, `'Imported'`), `recorded_at`, `system_prompt`, `user_prompt`, `model_output`, `original_transcript` (immutable), `rejected_output`, `rejected_model_version`, `rejection_reason`, 1–10 rubric scores, `normalized_prompt_hash`, `ngram_signature_json`, `leakage_status`, and `max_similarity_score`.
   - `leakage_audit_log`: Immutable audit log for manual near-duplicate overrides (`output_id`, `matched_benchmark_id`, `match_type`, `similarity_score`, `justification`, `overridden_at`).
4. **Evaluation Harness, Reproducibility & Dual Scoring (`eval_benchmarks`, `eval_runs`, `eval_results`)**:
   - `eval_runs`: Links `checkpoint_id` (can be Stage 0 Google Gemma, Stage 1 Abliterated, Stage 2 `Tinman-Lab/Tinman-gemma4-companion-merged`, any Stage 4 Amanda LoRA Adapter, or a Teacher Gemini baseline) and `baseline_run_id` (for regression diffing). Records full reproducibility metadata (`inference_engine`, `engine_version`, `hardware_info`, `quantization_settings_json`, `system_prompt_sha256`, `benchmark_suite_version`, `generation_params_json`, and explicit non-determinism notes).
   - `eval_results`: Stores `output_text`, independent `human_scores_json`, independent `judge_scores_json` (with concise verbatim-quoted evidence rationales), and `agreement_metrics_json` (`mad_score`, `bias_delta`, `divergence_flagged`).
