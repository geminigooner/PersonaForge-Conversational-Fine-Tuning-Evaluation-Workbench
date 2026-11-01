<div align="center">

💜 GEMINI CORPUS

Conversational Fine-Tuning & Behavioral Evaluation Workbench

From conversational outputs to custom-trained model behavior.

A local-first research workbench for collecting multi-model Gemini conversations, curating behavioral datasets, fine-tuning Gemma with LoRA/QLoRA, and evaluating model adaptations.

<br/>
<br/>

🌸 MULTI-MODEL CORPUS · DATASET CURATION · MODEL ADAPTATION · BEHAVIORAL EVALUATION 🌸

</div>

⸻

✨ Overview

Gemini Corpus is an experimental model-training workbench designed to turn conversational data from multiple Gemini models into curated training datasets for an existing Gemma checkpoint.

Rather than treating conversational outputs as disposable chat history, the project provides a structured environment for preserving, annotating, comparing, and transforming them into supervised fine-tuning and preference-optimization datasets.

The primary training target is:

Tinman-Lab/Tinman-gemma4-companion-merged

The workbench is designed to extend this existing modified Gemma checkpoint through new LoRA/QLoRA adapters, using curated conversational examples and preference comparisons.

The central idea

Collect outputs from different Gemini models, identify the behavioral characteristics worth preserving, prepare training examples, adapt Gemma, and evaluate what changes.

⸻

🧬 The Pipeline

<div align="center">
       MULTIPLE GEMINI MODELS
    ┌──────────┬──────────┬──────────┐
    │ Gemini   │ Gemini   │ Gemini   │
    │ 2.5 Pro  │ 3.1 Pro  │ 3 Flash  │
    └──────────┴──────────┴──────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │   SQLITE CORPUS      │
       │                      │
       │ Conversations        │
       │ Model Attribution    │
       │ Original Transcripts │
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ BEHAVIORAL ANNOTATION│
       │                      │
       │ Character Consistency│
       │ Narrative Continuity │
       │ Expressiveness       │
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │   DATASET STUDIO     │
       │                      │
       │ SFT Examples         │
       │ DPO Preference Pairs │
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │   TINMAN GEMMA       │
       │                      │
       │ LoRA / QLoRA         │
       │ SFT / DPO Training   │
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  ADAPTER CHECKPOINTS │
       └──────────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │ EVALUATION HARNESS   │
       │                      │
       │ Model Comparisons    │
       │ Behavioral Scoring   │
       │ Regression Testing   │
       └──────────────────────┘
</div>

⸻

🌷 Core Workspaces

Workspace	Purpose
💜 Dashboard	Corpus statistics, model activity, datasets, and experiments
🌸 Corpus Studio	Import, preserve, organize, and filter multi-model conversations
🎀 Annotation Studio	Score and annotate conversational behavior
🧁 Dataset Studio	Construct SFT examples and DPO preference pairs
🧬 Model Lab	Inspect checkpoints and configure LoRA/QLoRA
⚗️ Training Studio	Configure and manage fine-tuning experiments
📊 Evaluation Lab	Benchmark models and compare behavioral changes

01 — Multi-Model Conversation Corpus

The corpus is designed to support multiple Gemini models rather than a single fixed teacher model.

Initial source-model presets include:

* Gemini 2.5 Pro
* Gemini 3.1 Pro
* Gemini 3 Flash
* Additional Gemini variants
* Custom model identifiers

Conversations retain source-model attribution, provider information, timestamps, prompts, outputs, and available generation metadata.

The system is designed to support filtering, grouping, and comparing conversations by their originating model.

Individual assistant turns can also retain separate model attribution.

02 — Behavioral Annotation

Conversational outputs can be reviewed and scored across six initial behavioral dimensions:

Dimension	Description
Persona Consistency	Persistence of an established character or persona
Narrative Continuity	Ability to maintain coherent conversational context
Emotional Expressiveness	Range and quality of emotional expression
Improvisation	Creative generation and spontaneous development
Conversational Adaptability	Responsiveness to changing interaction contexts
Instruction Following	Adherence to requested behavior and constraints

Human annotations remain separate from automated evaluation scores.

03 — SFT & DPO Dataset Preparation

Two distinct dataset workflows support different training objectives.

Supervised Fine-Tuning (SFT)

Curate desirable conversational examples into structured prompt-and-response training records.

Direct Preference Optimization (DPO)

Construct preference pairs containing:

* Chosen response
* Rejected response
* Original prompt and context
* Optional preference rationale
* Source-model attribution

Cross-model preference comparisons are supported as a design objective. For example, a preferred response from Gemini 2.5 Pro can be paired against a less desirable response from Gemini 3 Flash.

Dataset exports preserve provenance and support model-specific or combined training datasets.

04 — Tinman Gemma Adaptation

The primary target checkpoint is:

Tinman-Lab/Tinman-gemma4-companion-merged

The project is designed to build on this existing checkpoint rather than automatically restarting from untouched Gemma weights.

Planned capabilities include:

* Checkpoint and tokenizer inspection
* Model architecture verification
* LoRA configuration
* QLoRA configuration
* SFT training configuration
* DPO training configuration
* Training-script generation
* Adapter checkpoint management
* Validation-loss monitoring

Training configurations must be verified against the actual checkpoint and installed training environment before execution.

05 — Behavioral Evaluation Harness

A dedicated evaluation harness will compare the original model against newly adapted checkpoints.

Comparison targets include:

* Original Google Gemma 4
* Abliterated Gemma checkpoint
* Existing Tinman merged checkpoint
* Newly trained LoRA adapters
* Gemini teacher-model reference outputs

Evaluation will track improvements and regressions across the six behavioral dimensions.

Human ratings and LLM-as-a-Judge scores remain independent evaluation signals.

⸻

🧬 Model Lineage

Reproducibility requires knowing exactly where every model and adapter originated.

The project tracks a five-stage model lineage:

STAGE 0
Original Google Gemma 4 E4B IT
          │
          ▼
STAGE 1
Abliterated Gemma 4 E4B
          │
          ▼
STAGE 2
Tinman Gemma
SFT / DPO / Merged Checkpoint
          │
          ▼
STAGE 3
Curated Conversational Datasets
SFT / DPO Configurations
Training Experiments
          │
          ▼
STAGE 4
New LoRA Adapter Checkpoints
Behavioral Evaluation Results

Each experiment is intended to preserve its parent checkpoint, dataset version, training configuration, environment metadata, and resulting adapter.

This makes it possible to trace an experimental result back to the exact data and configuration that produced it.

⸻

🗄️ Data Architecture

SQLite serves as the application’s structured persistence layer.

The planned relational architecture includes:

Database Entity	Responsibility
model_checkpoints	Base models, parent checkpoints, and lineage
conversations	Conversation-level records
outputs	Individual turns and source-model attribution
dataset_versions	Versioned SFT/DPO datasets
lora_configs	Adapter and optimization configurations
training_runs	Fine-tuning experiment records
adapter_checkpoints	Resulting adapter artifacts
eval_benchmarks	Reusable evaluation prompts
eval_runs	Reproducible benchmark executions
eval_results	Model outputs and evaluation scores
leakage_audit_log	Auditable dataset leakage overrides

Data Integrity

Planned safeguards include:

* Immutable original transcript preservation.
* Exact duplicate detection.
* Near-duplicate detection.
* Conversation-level leakage checks.
* Dataset versioning.
* Export provenance.
* Auditable manual overrides.
* Database backup and restoration.

The database is for structured corpus and experiment management. It is not a RAG or vector database.

⸻

🎨 Visual Identity

The workbench uses a playful, tactile, kawaii-inspired visual language.

The goal is to make a technically serious research environment feel expressive, colorful, and distinctly personal.

Color Palette

Color	Role
💜 Deep Violet	Primary interface color
🪻 Lavender	Secondary surfaces
🌸 Bubblegum Pink	Interactive accents
🍦 Warm Cream	Content panels
🍇 Deep Plum	Borders and typography
⭐ Soft Gold	Highlights and indicators

Design characteristics:

* Chunky rounded components.
* Thick deep-plum outlines.
* Soft offset shadows.
* Pill-shaped controls.
* Playful technical icons.
* Rounded typography.
* Subtle gradients and decorative details.

Purple is the dominant color. Pink is an accent.

The interface should feel like a colorful experimental AI laboratory, not a generic enterprise dashboard.

⸻

🛠️ Technology Stack

Layer	Technology
Frontend	React 19
Build Tool	Vite
Styling	Tailwind CSS v4
Database	SQLite
SQLite Driver	better-sqlite3
Gemini Integration	Google GenAI SDK
Training Target	Gemma 4 E4B / Tinman merged checkpoint
Fine-Tuning	SFT / DPO
Parameter-Efficient Adaptation	LoRA / QLoRA
Training Ecosystem	Transformers / PEFT / TRL / Unsloth, subject to checkpoint compatibility

The application is designed around local-first data management, with external model services used where appropriate.

Local model inference and training are separate from Gemini teacher-model and judge-model API integrations.

⸻

🔬 Research Direction

Beyond producing a fine-tuned conversational model, this project creates an environment for investigating how training data influences observable model behavior.

Potential research questions include:

* Do different Gemini models produce systematically different conversational patterns?
* Which behavioral characteristics are retained after fine-tuning?
* Does combining multiple teacher-model datasets improve or degrade behavioral consistency?
* How do SFT and DPO affect different behavioral dimensions?
* Can improvements in one dimension introduce regressions in another?
* How stable are observed behavioral changes across different evaluation prompts?

The evaluation harness is intended to support controlled comparisons rather than relying exclusively on subjective impressions.

⸻

🚧 Development Status

This project is actively in development.

Component	Status
React 19 / Vite application shell	🟢 Implemented baseline
SQLite database	🟡 Planned
Multi-model corpus	🟡 Planned
Behavioral annotation	🟡 Planned
SFT/DPO dataset studio	🟡 Planned
Checkpoint verification	🟡 Planned
LoRA/QLoRA configuration	🟡 Planned
Training integration	🟡 Planned
Evaluation harness	🟡 Planned
Experiment lineage	🟡 Planned

Status labels describe the documented implementation plan and should be updated as features are completed and verified.

⸻

🎯 Project Objectives

1. Build a structured, reusable conversational corpus.
2. Preserve source-model attribution across multiple Gemini models.
3. Convert curated conversations into SFT and DPO datasets.
4. Extend the existing Tinman Gemma checkpoint through parameter-efficient fine-tuning.
5. Track model lineage and training experiments.
6. Evaluate behavioral changes using reproducible benchmarks.
7. Build a research environment that makes conversational model adaptation inspectable and repeatable.

⸻

<div align="center">

💜 A LITTLE DATA. A LOT OF PERSONALITY. A WHOLE LOT OF EXPERIMENTATION.

Turning conversational behavior into training data, and training data into measurable model adaptations.

<br/>

Built with curiosity, questionable amounts of purple, and an unreasonable number of conversations. 🌸

</div>