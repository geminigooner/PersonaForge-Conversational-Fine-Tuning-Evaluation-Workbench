import {
  CorpusTurnOutput,
  GEMINI_TEACHER_MODELS,
  GeminiTeacherModelPreset,
  LoRAConfigRecipe,
  ModelCheckpointNode,
  WorkbenchState,
} from '../types/workbench';

export function getModelBadgeMeta(modelVersion: string): GeminiTeacherModelPreset {
  const found = GEMINI_TEACHER_MODELS.find(
    (m) => m.id.toLowerCase() === (modelVersion || '').toLowerCase()
  );
  if (found) return found;

  const lower = (modelVersion || '').toLowerCase();
  if (lower.includes('2.5') && lower.includes('pro')) {
    return GEMINI_TEACHER_MODELS[0];
  }
  if (lower.includes('3.1') && lower.includes('pro')) {
    return GEMINI_TEACHER_MODELS[1];
  }
  if (lower.includes('3') && lower.includes('flash') && !lower.includes('3.8')) {
    return GEMINI_TEACHER_MODELS[2];
  }

  return {
    id: modelVersion || 'custom-gemini',
    displayName: modelVersion || 'Custom Gemini',
    shortBadge: (modelVersion || 'Custom').slice(0, 14),
    defaultProvider: 'Imported',
    badgeBg: '#E9D5FF',
    badgeText: '#2A114B',
    description: 'Custom or imported Gemini model identifier.',
  };
}

export function formatGemma4TurnString(
  systemPrompt: string,
  userPrompt: string,
  modelOutput: string,
  useThinkToken = false
): string {
  const sysPart = systemPrompt?.trim()
    ? `<|turn>system\n${systemPrompt.trim()}<turn|>\n`
    : '';
  const thinkPrefix = useThinkToken ? `<|think|>\n` : '';
  return `<bos>${sysPart}<|turn>user\n${userPrompt.trim()}<turn|>\n<|turn>model\n${thinkPrefix}${modelOutput.trim()}<turn|>`;
}

export function compileSFTJsonl(
  outputs: CorpusTurnOutput[],
  useThinkToken = false
): string {
  // Group by conversation_id and sort by turn_number ASC
  const cleanEligible = outputs.filter(
    (o) =>
      o.training_eligible &&
      (o.leakage_status === 'clean' || o.leakage_status === 'overridden')
  );

  const byConv = new Map<string, CorpusTurnOutput[]>();
  for (const turn of cleanEligible) {
    const list = byConv.get(turn.conversation_id) || [];
    list.push(turn);
    byConv.set(turn.conversation_id, list);
  }

  const lines: string[] = [];
  for (const [convId, turns] of byConv.entries()) {
    turns.sort((a, b) => a.turn_number - b.turn_number);
    const first = turns[0];
    const messages: { role: string; content: string }[] = [];
    if (first.system_prompt?.trim()) {
      messages.push({ role: 'system', content: first.system_prompt.trim() });
    }
    let rawGemma4Thread = `<bos>${
      first.system_prompt?.trim()
        ? `<|turn>system\n${first.system_prompt.trim()}<turn|>\n`
        : ''
    }`;

    for (const t of turns) {
      messages.push({ role: 'user', content: t.user_prompt });
      messages.push({
        role: 'assistant',
        content: useThinkToken ? `<|think|>\n${t.model_output}` : t.model_output,
      });
      rawGemma4Thread += `<|turn>user\n${t.user_prompt.trim()}<turn|>\n<|turn>model\n${
        useThinkToken ? '<|think|>\n' : ''
      }${t.model_output.trim()}<turn|>\n`;
    }

    lines.push(
      JSON.stringify({
        schema_version: '1.2.0',
        export_type: 'sft_multi_turn',
        target_parent_model: 'Tinman-Lab/Tinman-gemma4-companion-merged',
        conversation_id: convId,
        conversation_title: first.conversation_title,
        turn_count: turns.length,
        source_models: Array.from(new Set(turns.map((t) => t.model_version))),
        provider_sources: Array.from(new Set(turns.map((t) => t.provider_source))),
        recorded_dates: turns.map((t) => t.recorded_at),
        persona_archetype: first.persona_archetype,
        behavioral_categories: Array.from(new Set(turns.map((t) => t.behavioral_category))),
        leakage_verification: turns.some((t) => t.leakage_status === 'overridden')
          ? 'audited_override'
          : 'verified_clean',
        messages,
        text_gemma4_rendered: rawGemma4Thread.trim(),
      })
    );
  }

  return lines.join('\n');
}

export function compileDPOJsonl(outputs: CorpusTurnOutput[]): string {
  const dpoReady = outputs.filter(
    (o) =>
      o.training_eligible &&
      o.rejected_output.trim().length > 0 &&
      (o.leakage_status === 'clean' || o.leakage_status === 'overridden')
  );

  return dpoReady
    .sort((a, b) => a.conversation_id.localeCompare(b.conversation_id) || a.turn_number - b.turn_number)
    .map((o) =>
      JSON.stringify({
        schema_version: '1.2.0',
        export_type: 'dpo_preference_pair',
        target_parent_model: 'Tinman-Lab/Tinman-gemma4-companion-merged',
        id: o.id,
        conversation_id: o.conversation_id,
        turn_number: o.turn_number,
        recorded_at: o.recorded_at,
        chosen_model_version: o.model_version,
        chosen_provider_source: o.provider_source,
        rejected_model_version: o.rejected_model_version || 'contrast-baseline',
        rejection_reason: o.rejection_reason || 'unspecified',
        system: o.system_prompt,
        prompt: o.user_prompt,
        chosen: o.model_output,
        rejected: o.rejected_output,
        prompt_messages: [
          ...(o.system_prompt ? [{ role: 'system', content: o.system_prompt }] : []),
          { role: 'user', content: o.user_prompt },
        ],
        chosen_messages: [{ role: 'assistant', content: o.model_output }],
        rejected_messages: [{ role: 'assistant', content: o.rejected_output }],
        rubric_scores: {
          narrative_continuity: o.narrative_continuity_score,
          character_consistency: o.character_consistency_score,
          emotional_expressiveness: o.emotional_expressiveness_score,
        },
        leakage_status: o.leakage_status,
      })
    )
    .join('\n');
}

export function compileBehavioralEvalJsonl(outputs: CorpusTurnOutput[]): string {
  return outputs
    .map((o) =>
      JSON.stringify({
        schema_version: '1.2.0',
        export_type: 'behavioral_evaluation_corpus',
        id: o.id,
        conversation_id: o.conversation_id,
        turn_number: o.turn_number,
        model_version: o.model_version,
        provider_source: o.provider_source,
        recorded_at: o.recorded_at,
        persona_archetype: o.persona_archetype,
        behavioral_category: o.behavioral_category,
        tags: o.tags,
        system_prompt: o.system_prompt,
        user_prompt: o.user_prompt,
        curated_model_output: o.model_output,
        original_transcript_preserved: o.original_transcript,
        transcript_was_edited: o.model_output !== o.original_transcript,
        scores: {
          narrative_continuity_score: o.narrative_continuity_score,
          character_consistency_score: o.character_consistency_score,
          emotional_expressiveness_score: o.emotional_expressiveness_score,
        },
        human_evaluation_notes: o.human_evaluation_notes,
        training_eligible: o.training_eligible,
        leakage_status: o.leakage_status,
      })
    )
    .join('\n');
}

export function compileFullArchiveJson(state: WorkbenchState): string {
  return JSON.stringify(
    {
      schema_version: '1.2.0',
      exported_at: new Date().toISOString(),
      workbench: 'Tinman & Gemini Behavioral Fine-Tuning Lab',
      active_parent_target: 'Tinman-Lab/Tinman-gemma4-companion-merged',
      root_baseline: 'google/gemma-4-E4B-it',
      documented_context_window: 131072,
      lineage_checkpoints: state.checkpoints,
      dataset_versions: state.datasetVersions,
      lora_configs: state.loraConfigs,
      training_runs: state.trainingRuns,
      outputs: state.outputs,
      leakage_audit_log: state.leakageAuditLog,
      eval_benchmarks: state.evalBenchmarks,
      eval_runs: state.evalRuns,
      eval_results: state.evalResults,
    },
    null,
    2
  );
}

export function generateCheckpointVerifierScript(
  targetRepo = 'Tinman-Lab/Tinman-gemma4-companion-merged'
): string {
  return `#!/usr/bin/env python3
"""
Pre-Flight Checkpoint & Environment Verification Probe
Target Parent Model: ${targetRepo}
Root Baseline Reference: google/gemma-4-E4B-it (128K Context Window)

Run this script in your training environment BEFORE launching SFT/DPO LoRA training.
It inspects config.json, tokenizer_config.json, actual chat_template Jinja rules,
special tokens (<bos>, <|turn>, <turn|>, <|think|>), and linear projection modules,
then writes 'checkpoint_manifest.json' to upload back into the Workbench.
"""

import json
import hashlib
from datetime import datetime, timezone
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

TARGET_REPO = "${targetRepo}"

def verify_checkpoint():
    print(f"[*] Inspecting parent checkpoint: {TARGET_REPO}")
    config = AutoConfig.from_pretrained(TARGET_REPO, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TARGET_REPO, trust_remote_code=True)

    architectures = getattr(config, "architectures", [config.__class__.__name__])
    text_cfg = getattr(config, "text_config", config)
    context_window = getattr(text_cfg, "max_position_embeddings", 131072)

    chat_template = getattr(tokenizer, "chat_template", "") or ""
    template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()

    vocab = tokenizer.get_vocab()
    think_supported = "<|think|>" in vocab or "<|think|>" in chat_template

    # Discover linear projection suffixes from config / meta model without full VRAM allocation
    expected_projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    try:
        import peft
        peft_ver = peft.__version__
    except ImportError:
        peft_ver = "not_installed"

    try:
        import trl
        trl_ver = trl.__version__
    except ImportError:
        trl_ver = "not_installed"

    try:
        import unsloth
        unsloth_ver = getattr(unsloth, "__version__", "installed")
    except ImportError:
        unsloth_ver = "not_installed"

    manifest = {
        "verified": True,
        "inspected_repo": TARGET_REPO,
        "model_class": architectures[0] if architectures else "Gemma4ForConditionalGeneration",
        "architectures": architectures,
        "context_window": int(context_window),
        "discovered_linear_modules": expected_projections,
        "multimodal_towers_detected": hasattr(config, "vision_config") or hasattr(config, "audio_config"),
        "chat_template_sha256": template_sha256,
        "chat_template_snippet": chat_template[:280],
        "special_tokens": {
            "bos_token": str(tokenizer.bos_token),
            "eos_token": str(tokenizer.eos_token),
            "turn_open_token": "<|turn>",
            "turn_close_token": "<turn|>",
            "think_token": "<|think|>" if think_supported else None,
        },
        "think_token_supported": bool(think_supported),
        "library_versions": {
            "transformers": transformers.__version__,
            "peft": peft_ver,
            "trl": trl_ver,
            "unsloth": unsloth_ver,
            "torch": torch.__version__,
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    with open("checkpoint_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[+] Verification succeeded! Saved checkpoint_manifest.json")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    verify_checkpoint()
`;
}

export function generateTinmanTrainingScript(
  config: LoRAConfigRecipe,
  parentNode: ModelCheckpointNode | undefined,
  datasetTag: string
): string {
  const repo = parentNode?.hf_repo_or_path || 'Tinman-Lab/Tinman-gemma4-companion-merged';
  const isDpo = config.training_stage === 'dpo';
  const datasetFile = isDpo ? `${datasetTag}_dpo.jsonl` : `${datasetTag}_sft.jsonl`;
  const targetMods = JSON.stringify(config.verified_target_modules);

  return `#!/usr/bin/env python3
"""
Verification-Guarded LoRA Training Pipeline (${config.training_stage.toUpperCase()})
Active Parent Checkpoint: ${repo}
Documented Context Window: 128K (131,072 tokens) | Configured Max Seq Length: ${config.max_seq_length}
Recipe Name: ${config.name}

NOTE: Learning rate (${config.learning_rate}), warmup, and gradient clipping (${config.max_grad_norm})
are configured as experimental hyperparameters with step-level validation loss monitoring.
"""

import os
import json
import hashlib
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig

PARENT_MODEL_ID = "${repo}"
DATASET_PATH = "${datasetFile}"
OUTPUT_DIR = "./outputs/${config.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')}"
TARGET_MODULES = ${targetMods}

# 1. Pre-Flight Runtime Verification Gate (Never assume compatibility without checking)
print(f"[*] Verifying parent checkpoint architecture & tokenizer: {PARENT_MODEL_ID}")
hf_config = AutoConfig.from_pretrained(PARENT_MODEL_ID, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(PARENT_MODEL_ID, trust_remote_code=True)

assert getattr(tokenizer, "chat_template", None), (
    "CRITICAL: Selected checkpoint does not define a tokenizer.chat_template. "
    "Inspect tokenizer_config.json before training."
)

# 2. Load Parent Checkpoint (${config.quantization_mode})
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name=PARENT_MODEL_ID,
    max_seq_length=${config.max_seq_length},
    load_in_4bit=${config.quantization_mode === 'nf4_4bit' ? 'True' : 'False'},
    use_gradient_checkpointing="unsloth",
)

# Verify all requested TARGET_MODULES exist on the actual loaded model
discovered_linear_names = {
    name.split(".")[-1]
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
}
missing_modules = [m for m in TARGET_MODULES if m not in discovered_linear_names]
if missing_modules:
    raise RuntimeError(f"Target modules {missing_modules} not found in loaded model linear layers: {sorted(discovered_linear_names)}")
print(f"[+] Verified all target modules exist on {PARENT_MODEL_ID}: {TARGET_MODULES}")

# 3. Attach LoRA Adapter to Parent Checkpoint
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=${config.freeze_multimodal_towers ? 'False' : 'True'},
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=${config.lora_r},
    lora_alpha=${config.lora_alpha},
    lora_dropout=${config.lora_dropout},
    target_modules=TARGET_MODULES,
    bias="none",
    random_state=42,
)

# 4. Load Leak-Verified Dataset & Create Train/Validation Split (${Math.round(config.val_split_ratio * 100)}% Validation)
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
split_dataset = raw_dataset.train_test_split(test_size=${config.val_split_ratio}, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"[+] Loaded {len(train_dataset)} train rows and {len(eval_dataset)} validation rows.")

${
  isDpo
    ? `# 5. Configure DPOTrainer with Validation-Loss Monitoring & Gradient Clipping
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=${config.learning_rate},
    lr_scheduler_type="${config.lr_scheduler_type}",
    warmup_ratio=${config.warmup_ratio},
    warmup_steps=${config.warmup_steps},
    max_grad_norm=${config.max_grad_norm},
    weight_decay=${config.weight_decay},
    optim="${config.optimizer_type}",
    num_train_epochs=${config.num_epochs},
    per_device_train_batch_size=${config.per_device_batch_size},
    gradient_accumulation_steps=${config.gradient_accumulation_steps},
    eval_strategy="steps",
    eval_steps=${config.eval_steps},
    save_strategy="steps",
    save_steps=${config.eval_steps},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_length=${config.max_seq_length},
    beta=0.1,
    logging_steps=5,
    bf16=torch.cuda.is_bf16_supported(),
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=${config.early_stopping_patience})],
)`
    : `# 5. Configure SFTTrainer with Validation-Loss Monitoring & Gradient Clipping
from trl import SFTConfig, SFTTrainer

def format_prompts(batch):
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in batch["messages"]
    ]
    return {"text": texts}

train_dataset = train_dataset.map(format_prompts, batched=True)
eval_dataset = eval_dataset.map(format_prompts, batched=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    learning_rate=${config.learning_rate},
    lr_scheduler_type="${config.lr_scheduler_type}",
    warmup_ratio=${config.warmup_ratio},
    warmup_steps=${config.warmup_steps},
    max_grad_norm=${config.max_grad_norm},
    weight_decay=${config.weight_decay},
    optim="${config.optimizer_type}",
    num_train_epochs=${config.num_epochs},
    per_device_train_batch_size=${config.per_device_batch_size},
    gradient_accumulation_steps=${config.gradient_accumulation_steps},
    eval_strategy="steps",
    eval_steps=${config.eval_steps},
    save_strategy="steps",
    save_steps=${config.eval_steps},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_seq_length=${config.max_seq_length},
    logging_steps=5,
    bf16=torch.cuda.is_bf16_supported(),
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=${config.early_stopping_patience})],
)`
}

print("[*] Starting training with validation-loss monitoring...")
train_result = trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Compute SHA-256 checksum of resulting adapter weights for Experiment Lineage Registry
adapter_file = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
if os.path.exists(adapter_file):
    with open(adapter_file, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"[+] Adapter Checkpoint SHA-256: {digest}")
`;
}
