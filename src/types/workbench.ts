export type BehavioralDimension =
  | 'persona_consistency'
  | 'narrative_continuity'
  | 'emotional_expressiveness'
  | 'improvisation'
  | 'conversational_adaptability'
  | 'instruction_following';

export const BEHAVIORAL_DIMENSIONS: {
  key: BehavioralDimension;
  label: string;
  shortLabel: string;
  description: string;
}[] = [
  {
    key: 'persona_consistency',
    label: 'Persona Consistency',
    shortLabel: 'Persona',
    description: 'Adherence to established character voice, boundaries, and archetype across turns.',
  },
  {
    key: 'narrative_continuity',
    label: 'Narrative Continuity',
    shortLabel: 'Continuity',
    description: 'Retention of prior scene context, spatial/temporal details, and causal flow.',
  },
  {
    key: 'emotional_expressiveness',
    label: 'Emotional Expressiveness',
    shortLabel: 'Expressiveness',
    description: 'Affective warmth, nuanced subtext, and authentic emotional cadence.',
  },
  {
    key: 'improvisation',
    label: 'Improvisation',
    shortLabel: 'Improvisation',
    description: 'Creative expansion on novel conversational hooks without repetitive stalling.',
  },
  {
    key: 'conversational_adaptability',
    label: 'Conversational Adaptability',
    shortLabel: 'Adaptability',
    description: 'Graceful handling of sudden tone shifts, ambiguity, or user pushback.',
  },
  {
    key: 'instruction_following',
    label: 'Instruction Following',
    shortLabel: 'Instructions',
    description: 'Strict compliance with structural constraints, negative prompts, and turn formatting.',
  },
];

export interface GeminiTeacherModelPreset {
  id: string;
  displayName: string;
  shortBadge: string;
  defaultProvider: 'OpenRouter' | 'AI Studio' | 'Live API' | 'Imported';
  badgeBg: string;
  badgeText: string;
  description: string;
}

export const GEMINI_TEACHER_MODELS: GeminiTeacherModelPreset[] = [
  {
    id: 'google/gemini-2.5-pro',
    displayName: 'Gemini 2.5 Pro',
    shortBadge: 'Pro 2.5',
    defaultProvider: 'OpenRouter',
    badgeBg: '#FF5CA8',
    badgeText: '#FFFFFF',
    description: 'Primary OpenRouter teacher model for deep narrative continuity and nuanced persona transcripts.',
  },
  {
    id: 'gemini-3.1-pro-preview',
    displayName: 'Gemini 3.1 Pro',
    shortBadge: 'Pro 3.1',
    defaultProvider: 'AI Studio',
    badgeBg: '#7C3AED',
    badgeText: '#FFFFFF',
    description: 'High-reasoning Gemini 3.1 Pro teacher checkpoint for complex multi-turn instruction alignment.',
  },
  {
    id: 'gemini-3-flash',
    displayName: 'Gemini 3 Flash',
    shortBadge: 'Flash 3',
    defaultProvider: 'AI Studio',
    badgeBg: '#FACC15',
    badgeText: '#2A114B',
    description: 'Fast conversational Gemini 3 Flash teacher model for high-throughput dialogue turns and DPO contrasts.',
  },
  {
    id: 'gemini-3.8-flash',
    displayName: 'Gemini 3.8 Flash (Live SDK)',
    shortBadge: 'Flash 3.8',
    defaultProvider: 'Live API',
    badgeBg: '#6EE7B7',
    badgeText: '#065F46',
    description: 'Live server-side @google/genai SDK teacher & contrast drafter.',
  },
  {
    id: 'gemini-4-argon',
    displayName: 'Gemini 4 Argon (Live SDK)',
    shortBadge: 'Argon 4',
    defaultProvider: 'Live API',
    badgeBg: '#C084FC',
    badgeText: '#2A114B',
    description: 'Frontier Gemini 4 Argon teacher & automated rubric judge.',
  },
];

export type LeakageStatus =
  | 'clean'
  | 'near_duplicate_warn'
  | 'conversation_subsequence_blocked'
  | 'exact_collision_blocked'
  | 'overridden';

export interface CorpusTurnOutput {
  id: string;
  conversation_id: string;
  conversation_title: string;
  turn_number: number;
  model_version: string;
  provider_source: string;
  recorded_at: string;
  system_prompt: string;
  user_prompt: string;
  model_output: string;
  original_transcript: string;
  full_context_json: string;
  rejected_output: string;
  rejected_model_version: string;
  rejection_reason: string;
  behavioral_category: string;
  tags: string[];
  persona_archetype: string;
  narrative_continuity_score: number;
  character_consistency_score: number;
  emotional_expressiveness_score: number;
  human_evaluation_notes: string;
  training_eligible: boolean;
  source_mode: 'manual' | 'live_api' | 'imported';
  normalized_prompt_hash: string;
  leakage_status: LeakageStatus;
  max_similarity_score: number;
  matched_benchmark_id: string | null;
  override_justification?: string | null;
}

export interface LeakageAuditRecord {
  id: string;
  output_id: string;
  conversation_id: string;
  matched_benchmark_id: string;
  matched_benchmark_title: string;
  match_type: string;
  similarity_score: number;
  justification: string;
  operator_name: string;
  overridden_at: string;
}

export interface VerifiedCheckpointManifest {
  verified: boolean;
  inspected_repo: string;
  model_class: string;
  architectures: string[];
  context_window: number;
  discovered_linear_modules: string[];
  multimodal_towers_detected: boolean;
  chat_template_sha256: string;
  chat_template_snippet: string;
  special_tokens: {
    bos_token: string;
    eos_token: string;
    turn_open_token: string;
    turn_close_token: string;
    think_token: string | null;
  };
  think_token_supported: boolean;
  library_versions: {
    transformers: string;
    peft: string;
    trl: string;
    unsloth: string;
    torch: string;
  };
  verified_at: string;
}

export type LineageStage =
  | 'stage_0_root'
  | 'stage_1_abliterated'
  | 'stage_2_tinman_merged'
  | 'stage_4_lora_adapter'
  | 'teacher_reference';

export interface ModelCheckpointNode {
  id: string;
  parent_checkpoint_id: string | null;
  stage: LineageStage;
  name: string;
  hf_repo_or_path: string;
  revision_commit_sha: string;
  weights_sha256: string;
  context_window: number;
  verified_manifest: VerifiedCheckpointManifest;
  provenance_notes: string;
  created_at: string;
}

export interface DatasetVersionSnapshot {
  id: string;
  version_tag: string;
  split_type: 'sft' | 'dpo';
  teacher_models_filter: string[];
  teacher_model_breakdown: Record<string, number>;
  train_count: number;
  val_count: number;
  dataset_sha256: string;
  leakage_status: 'verified_clean' | 'contains_audited_overrides';
  override_count: number;
  created_at: string;
}

export interface LoRAConfigRecipe {
  id: string;
  name: string;
  parent_checkpoint_id: string;
  training_stage: 'sft' | 'dpo';
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  verified_target_modules: string[];
  freeze_multimodal_towers: boolean;
  learning_rate: number;
  lr_scheduler_type: 'cosine' | 'linear' | 'constant_with_warmup';
  warmup_ratio: number;
  warmup_steps: number;
  max_grad_norm: number;
  weight_decay: number;
  optimizer_type: 'adamw_8bit' | 'paged_adamw_32bit' | 'adamw_torch';
  num_epochs: number;
  per_device_batch_size: number;
  gradient_accumulation_steps: number;
  val_split_ratio: number;
  eval_steps: number;
  early_stopping_patience: number;
  max_seq_length: number;
  quantization_mode: 'nf4_4bit' | 'bf16_16bit' | 'qat_q4_0';
  use_think_control_token: boolean;
  created_at: string;
}

export interface TrainingRunRecord {
  id: string;
  run_title: string;
  parent_checkpoint_id: string;
  dataset_version_id: string;
  lora_config_id: string;
  resulting_adapter_checkpoint_id: string;
  status: 'completed' | 'running_external' | 'planned';
  train_loss_curve: { step: number; loss: number }[];
  val_loss_curve: { step: number; val_loss: number }[];
  best_val_loss: number;
  final_train_loss: number;
  environment_versions: Record<string, string>;
  notes: string;
  started_at: string;
  completed_at: string;
}

export interface EvalBenchmarkPrompt {
  id: string;
  suite_version: string;
  title: string;
  category: BehavioralDimension;
  system_prompt: string;
  prior_context: { role: 'user' | 'model'; content: string }[];
  user_prompt: string;
  evaluation_criteria: string;
  reference_anchor: string;
  normalized_prompt_hash: string;
  created_at: string;
}

export interface DimensionalScores {
  persona_consistency: number;
  narrative_continuity: number;
  emotional_expressiveness: number;
  improvisation: number;
  conversational_adaptability: number;
  instruction_following: number;
}

export interface HumanScorePayload extends DimensionalScores {
  rater_notes: string;
  rated_at: string;
}

export interface JudgeScorePayload extends DimensionalScores {
  judge_model_id: string;
  judge_rubric_version: string;
  evidence_rationales: Record<BehavioralDimension, string>;
  overall_summary: string;
  judged_at: string;
}

export interface AgreementMetrics {
  mad_score: number;
  max_dimension_delta: number;
  bias_delta: number; // Judge mean minus Human mean
  per_dimension_delta: Record<BehavioralDimension, number>;
  divergence_flagged: boolean; // True if any dimension differs by >= 2.0
}

export interface EvalRunRecord {
  id: string;
  run_name: string;
  checkpoint_id: string;
  baseline_run_id: string | null;
  benchmark_suite_version: string;
  execution_source: 'local_endpoint' | 'batch_jsonl_import' | 'teacher_reference';
  inference_engine: string;
  engine_version: string;
  hardware_info: string;
  quantization_settings: {
    method: string;
    bits: number;
    compute_dtype: string;
    kv_cache_dtype: string;
  };
  system_prompt_sha256: string;
  generation_params: {
    temperature: number;
    top_p: number;
    top_k: number;
    min_p: number;
    repetition_penalty: number;
    max_new_tokens: number;
    seed: number;
  };
  determinism_notes: string;
  created_at: string;
}

export interface EvalResultItem {
  id: string;
  run_id: string;
  benchmark_id: string;
  output_text: string;
  output_sha256: string;
  latency_ms: number;
  token_count: number;
  human_scores: HumanScorePayload | null;
  judge_scores: JudgeScorePayload | null;
  agreement_metrics: AgreementMetrics | null;
}

export interface WorkbenchState {
  checkpoints: ModelCheckpointNode[];
  outputs: CorpusTurnOutput[];
  leakageAuditLog: LeakageAuditRecord[];
  datasetVersions: DatasetVersionSnapshot[];
  loraConfigs: LoRAConfigRecipe[];
  trainingRuns: TrainingRunRecord[];
  evalBenchmarks: EvalBenchmarkPrompt[];
  evalRuns: EvalRunRecord[];
  evalResults: EvalResultItem[];
}
