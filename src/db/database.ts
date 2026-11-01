import Database from 'better-sqlite3';
import crypto from 'crypto';
import path from 'path';
import {
  AgreementMetrics,
  BehavioralDimension,
  BEHAVIORAL_DIMENSIONS,
  CorpusTurnOutput,
  DatasetVersionSnapshot,
  EvalBenchmarkPrompt,
  EvalResultItem,
  EvalRunRecord,
  HumanScorePayload,
  JudgeScorePayload,
  LeakageAuditRecord,
  LeakageStatus,
  LoRAConfigRecipe,
  ModelCheckpointNode,
  TrainingRunRecord,
  VerifiedCheckpointManifest,
  WorkbenchState,
} from '../types/workbench';

const dbPath = path.resolve(process.cwd(), 'tinman_gemini_workbench.db');
export const db = new Database(dbPath);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

export function sha256Hex(input: string): string {
  return crypto.createHash('sha256').update(input, 'utf8').digest('hex');
}

export function normalizePromptText(input: string): string {
  return input
    .toLowerCase()
    .replace(/<\|?turn\|?>|<bos>|<eos>|<\|think\|>/g, ' ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function computeNGramTokens(input: string): string[] {
  const norm = normalizePromptText(input);
  if (!norm) return [];
  const words = norm.split(' ').filter(Boolean);
  const grams = new Set<string>();
  for (let i = 0; i < words.length; i++) {
    grams.add(words[i]);
    if (i + 1 < words.length) grams.add(`${words[i]}_${words[i + 1]}`);
    if (i + 2 < words.length) grams.add(`${words[i]}_${words[i + 1]}_${words[i + 2]}`);
  }
  const compact = norm.replace(/\s+/g, '');
  for (let i = 0; i <= compact.length - 4; i += 2) {
    grams.add(`c4_${compact.slice(i, i + 4)}`);
  }
  return Array.from(grams);
}

export function computeJaccardSimilarity(aGrams: string[], bGrams: string[]): number {
  if (aGrams.length === 0 || bGrams.length === 0) return 0;
  const setA = new Set(aGrams);
  const setB = new Set(bGrams);
  let intersection = 0;
  for (const item of setA) {
    if (setB.has(item)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : Number((intersection / union).toFixed(4));
}

export function calculateAgreementMetrics(
  human: HumanScorePayload | null,
  judge: JudgeScorePayload | null
): AgreementMetrics | null {
  if (!human || !judge) return null;
  const dims = BEHAVIORAL_DIMENSIONS.map((d) => d.key);
  let absDiffSum = 0;
  let signedDiffSum = 0;
  let maxDelta = 0;
  const perDimDelta = {} as Record<BehavioralDimension, number>;

  for (const dim of dims) {
    const hVal = Number(human[dim] ?? 0);
    const jVal = Number(judge[dim] ?? 0);
    const signed = Number((jVal - hVal).toFixed(2));
    const abs = Math.abs(signed);
    perDimDelta[dim] = signed;
    absDiffSum += abs;
    signedDiffSum += signed;
    if (abs > maxDelta) maxDelta = abs;
  }

  const madScore = Number((absDiffSum / dims.length).toFixed(2));
  const biasDelta = Number((signedDiffSum / dims.length).toFixed(2));
  return {
    mad_score: madScore,
    max_dimension_delta: Number(maxDelta.toFixed(2)),
    bias_delta: biasDelta,
    per_dimension_delta: perDimDelta,
    divergence_flagged: maxDelta >= 2.0,
  };
}

export function evaluateFourLayerLeakage(
  userPrompt: string,
  fullContextJson: string,
  benchmarks: { id: string; title: string; user_prompt: string; normalized_prompt_hash: string; ngram_json: string }[],
  hasAuditOverride = false
): {
  normalizedHash: string;
  ngrams: string[];
  status: LeakageStatus;
  maxSimilarity: number;
  matchedBenchmarkId: string | null;
} {
  const normPrompt = normalizePromptText(userPrompt);
  const normalizedHash = sha256Hex(normPrompt);
  const ngrams = computeNGramTokens(userPrompt);

  let contextTurns: { role?: string; content?: string }[] = [];
  try {
    contextTurns = JSON.parse(fullContextJson || '[]');
  } catch {
    contextTurns = [];
  }
  const contextHashes = new Set(
    contextTurns
      .map((t) => normalizePromptText(String(t.content || '')))
      .filter(Boolean)
      .map((txt) => sha256Hex(txt))
  );

  let maxSimilarity = 0;
  let matchedBenchmarkId: string | null = null;
  let exactCollision = false;
  let subsequenceCollision = false;

  for (const bm of benchmarks) {
    if (bm.normalized_prompt_hash === normalizedHash) {
      exactCollision = true;
      maxSimilarity = 1.0;
      matchedBenchmarkId = bm.id;
      break;
    }
    if (contextHashes.has(bm.normalized_prompt_hash)) {
      subsequenceCollision = true;
      maxSimilarity = Math.max(maxSimilarity, 0.95);
      matchedBenchmarkId = bm.id;
    }
    let bmGrams: string[] = [];
    try {
      bmGrams = JSON.parse(bm.ngram_json || '[]');
    } catch {
      bmGrams = computeNGramTokens(bm.user_prompt);
    }
    const sim = computeJaccardSimilarity(ngrams, bmGrams);
    if (sim > maxSimilarity) {
      maxSimilarity = sim;
      if (sim >= 0.38) {
        matchedBenchmarkId = bm.id;
      }
    }
  }

  if (exactCollision) {
    return {
      normalizedHash,
      ngrams,
      status: 'exact_collision_blocked',
      maxSimilarity: 1.0,
      matchedBenchmarkId,
    };
  }
  if (subsequenceCollision) {
    return {
      normalizedHash,
      ngrams,
      status: hasAuditOverride ? 'overridden' : 'conversation_subsequence_blocked',
      maxSimilarity,
      matchedBenchmarkId,
    };
  }
  if (maxSimilarity >= 0.42) {
    return {
      normalizedHash,
      ngrams,
      status: hasAuditOverride ? 'overridden' : 'near_duplicate_warn',
      maxSimilarity,
      matchedBenchmarkId,
    };
  }

  return {
    normalizedHash,
    ngrams,
    status: 'clean',
    maxSimilarity,
    matchedBenchmarkId: null,
  };
}

export function initDatabase() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS model_checkpoints (
      id TEXT PRIMARY KEY,
      parent_checkpoint_id TEXT,
      stage TEXT NOT NULL,
      name TEXT NOT NULL,
      hf_repo_or_path TEXT NOT NULL,
      revision_commit_sha TEXT NOT NULL,
      weights_sha256 TEXT NOT NULL,
      context_window INTEGER NOT NULL DEFAULT 131072,
      verified_manifest_json TEXT NOT NULL,
      provenance_notes TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS eval_benchmarks (
      id TEXT PRIMARY KEY,
      suite_version TEXT NOT NULL,
      title TEXT NOT NULL,
      category TEXT NOT NULL,
      system_prompt TEXT NOT NULL,
      prior_context_json TEXT NOT NULL,
      user_prompt TEXT NOT NULL,
      evaluation_criteria TEXT NOT NULL,
      reference_anchor TEXT NOT NULL,
      normalized_prompt_hash TEXT NOT NULL,
      ngram_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS outputs (
      id TEXT PRIMARY KEY,
      conversation_id TEXT NOT NULL,
      conversation_title TEXT NOT NULL,
      turn_number INTEGER NOT NULL,
      model_version TEXT NOT NULL,
      provider_source TEXT NOT NULL,
      recorded_at TEXT NOT NULL,
      system_prompt TEXT NOT NULL,
      user_prompt TEXT NOT NULL,
      model_output TEXT NOT NULL,
      original_transcript TEXT NOT NULL,
      full_context_json TEXT NOT NULL,
      rejected_output TEXT NOT NULL DEFAULT '',
      rejected_model_version TEXT NOT NULL DEFAULT '',
      rejection_reason TEXT NOT NULL DEFAULT '',
      behavioral_category TEXT NOT NULL,
      tags_json TEXT NOT NULL,
      persona_archetype TEXT NOT NULL,
      narrative_continuity_score REAL NOT NULL,
      character_consistency_score REAL NOT NULL,
      emotional_expressiveness_score REAL NOT NULL,
      human_evaluation_notes TEXT NOT NULL,
      training_eligible INTEGER NOT NULL DEFAULT 1,
      source_mode TEXT NOT NULL DEFAULT 'manual',
      normalized_prompt_hash TEXT NOT NULL,
      ngram_json TEXT NOT NULL,
      leakage_status TEXT NOT NULL DEFAULT 'clean',
      max_similarity_score REAL NOT NULL DEFAULT 0,
      matched_benchmark_id TEXT
    );

    CREATE TABLE IF NOT EXISTS leakage_audit_log (
      id TEXT PRIMARY KEY,
      output_id TEXT NOT NULL,
      conversation_id TEXT NOT NULL,
      matched_benchmark_id TEXT NOT NULL,
      matched_benchmark_title TEXT NOT NULL,
      match_type TEXT NOT NULL,
      similarity_score REAL NOT NULL,
      justification TEXT NOT NULL,
      operator_name TEXT NOT NULL,
      overridden_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dataset_versions (
      id TEXT PRIMARY KEY,
      version_tag TEXT NOT NULL,
      split_type TEXT NOT NULL,
      teacher_models_filter_json TEXT NOT NULL,
      teacher_model_breakdown_json TEXT NOT NULL,
      train_count INTEGER NOT NULL,
      val_count INTEGER NOT NULL,
      dataset_sha256 TEXT NOT NULL,
      leakage_status TEXT NOT NULL,
      override_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS lora_configs (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      parent_checkpoint_id TEXT NOT NULL,
      training_stage TEXT NOT NULL,
      lora_r INTEGER NOT NULL,
      lora_alpha INTEGER NOT NULL,
      lora_dropout REAL NOT NULL,
      verified_target_modules_json TEXT NOT NULL,
      freeze_multimodal_towers INTEGER NOT NULL DEFAULT 1,
      learning_rate REAL NOT NULL,
      lr_scheduler_type TEXT NOT NULL,
      warmup_ratio REAL NOT NULL,
      warmup_steps INTEGER NOT NULL,
      max_grad_norm REAL NOT NULL,
      weight_decay REAL NOT NULL,
      optimizer_type TEXT NOT NULL,
      num_epochs INTEGER NOT NULL,
      per_device_batch_size INTEGER NOT NULL,
      gradient_accumulation_steps INTEGER NOT NULL,
      val_split_ratio REAL NOT NULL,
      eval_steps INTEGER NOT NULL,
      early_stopping_patience INTEGER NOT NULL,
      max_seq_length INTEGER NOT NULL,
      quantization_mode TEXT NOT NULL,
      use_think_control_token INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS training_runs (
      id TEXT PRIMARY KEY,
      run_title TEXT NOT NULL,
      parent_checkpoint_id TEXT NOT NULL,
      dataset_version_id TEXT NOT NULL,
      lora_config_id TEXT NOT NULL,
      resulting_adapter_checkpoint_id TEXT NOT NULL,
      status TEXT NOT NULL,
      train_loss_curve_json TEXT NOT NULL,
      val_loss_curve_json TEXT NOT NULL,
      best_val_loss REAL NOT NULL,
      final_train_loss REAL NOT NULL,
      environment_versions_json TEXT NOT NULL,
      notes TEXT NOT NULL,
      started_at TEXT NOT NULL,
      completed_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS eval_runs (
      id TEXT PRIMARY KEY,
      run_name TEXT NOT NULL,
      checkpoint_id TEXT NOT NULL,
      baseline_run_id TEXT,
      benchmark_suite_version TEXT NOT NULL,
      execution_source TEXT NOT NULL,
      inference_engine TEXT NOT NULL,
      engine_version TEXT NOT NULL,
      hardware_info TEXT NOT NULL,
      quantization_settings_json TEXT NOT NULL,
      system_prompt_sha256 TEXT NOT NULL,
      generation_params_json TEXT NOT NULL,
      determinism_notes TEXT NOT NULL,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS eval_results (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      benchmark_id TEXT NOT NULL,
      output_text TEXT NOT NULL,
      output_sha256 TEXT NOT NULL,
      latency_ms INTEGER NOT NULL,
      token_count INTEGER NOT NULL,
      human_scores_json TEXT,
      judge_scores_json TEXT,
      agreement_metrics_json TEXT
    );
  `);

  const countRow = db.prepare('SELECT COUNT(*) as cnt FROM model_checkpoints').get() as { cnt: number };
  if (countRow.cnt === 0) {
    seedInitialData();
  }
}

function seedInitialData() {
  const defaultManifest = (
    repo: string,
    verified: boolean
  ): VerifiedCheckpointManifest => ({
    verified,
    inspected_repo: repo,
    model_class: 'Gemma4ForConditionalGeneration',
    architectures: ['Gemma4ForConditionalGeneration'],
    context_window: 131072,
    discovered_linear_modules: [
      'q_proj',
      'k_proj',
      'v_proj',
      'o_proj',
      'gate_proj',
      'up_proj',
      'down_proj',
    ],
    multimodal_towers_detected: true,
    chat_template_sha256: sha256Hex(`{{ bos_token }}{% for message in messages %}<|turn>{{ message['role'] }}\n{{ message['content'] }}<turn|>\n{% endfor %}`),
    chat_template_snippet:
      "{{ bos_token }}{% for message in messages %}<|turn>{{ message['role'] }}\\n{{ message['content'] }}<turn|>\\n{% endfor %}{% if add_generation_prompt %}<|turn>model\\n{% endif %}",
    special_tokens: {
      bos_token: '<bos>',
      eos_token: '<turn|>',
      turn_open_token: '<|turn>',
      turn_close_token: '<turn|>',
      think_token: '<|think|>',
    },
    think_token_supported: true,
    library_versions: {
      transformers: '4.51.2',
      peft: '0.15.1',
      trl: '0.16.1',
      unsloth: '2026.4.1',
      torch: '2.6.0+cu124',
    },
    verified_at: '2026-10-30T18:15:00Z',
  });

  const insertCkpt = db.prepare(`
    INSERT INTO model_checkpoints (
      id, parent_checkpoint_id, stage, name, hf_repo_or_path, revision_commit_sha,
      weights_sha256, context_window, verified_manifest_json, provenance_notes, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  insertCkpt.run(
    'ckpt_stage0_gemma4_e4b',
    null,
    'stage_0_root',
    'Google Gemma 4 E4B IT (Root Baseline)',
    'google/gemma-4-E4B-it',
    'a91c4e82f01d77b43e2109c8f11e5d3a782019ef',
    sha256Hex('google/gemma-4-E4B-it-weights-v1'),
    131072,
    JSON.stringify(defaultManifest('google/gemma-4-E4B-it', true)),
    'Original untouched Google Gemma 4 E4B IT checkpoint (4.4B effective parameters, Per-Layer Embeddings, 128K context window). Preserved strictly as Stage 0 root baseline.',
    '2026-10-12T09:00:00Z'
  );

  insertCkpt.run(
    'ckpt_stage1_abliterated',
    'ckpt_stage0_gemma4_e4b',
    'stage_1_abliterated',
    'Abliterated Gemma 4 E4B Checkpoint',
    'local-lineage/gemma-4-E4B-it-abliterated',
    'b42d991c78e34f01928374655a1b2c3d4e5f6071',
    sha256Hex('gemma-4-E4B-it-abliterated-weights'),
    131072,
    JSON.stringify(defaultManifest('local-lineage/gemma-4-E4B-it-abliterated', true)),
    'Stage 1 intermediate checkpoint derived from google/gemma-4-E4B-it via refusal-direction orthogonalization prior to Tinman companion alignment.',
    '2026-10-18T14:20:00Z'
  );

  insertCkpt.run(
    'ckpt_stage2_tinman_merged',
    'ckpt_stage1_abliterated',
    'stage_2_tinman_merged',
    'Tinman Gemma 4 Companion Merged (Active Parent Target)',
    'Tinman-Lab/Tinman-gemma4-companion-merged',
    'f78e310a94c25d88123b90cf441a8e729103c55b',
    sha256Hex('Tinman-Lab/Tinman-gemma4-companion-merged-bf16'),
    131072,
    JSON.stringify(defaultManifest('Tinman-Lab/Tinman-gemma4-companion-merged', true)),
    'Stage 2 active parent training target. Combines Tinman Lab SFT + DPO companion alignment merged into full-precision weights. All new Amanda LoRA experiments load this checkpoint as parent.',
    '2026-10-24T19:45:00Z'
  );

  insertCkpt.run(
    'ckpt_stage4_amanda_sft_v1',
    'ckpt_stage2_tinman_merged',
    'stage_4_lora_adapter',
    'Amanda LoRA Exp-01: Pure Pro 2.5 SFT (r=32)',
    'local-adapters/tinman-gemma4-amanda-sft-v1',
    'c01928374655fae1234567890abcdef123456789',
    sha256Hex('amanda-tinman-sft-v1-adapter-safetensors'),
    131072,
    JSON.stringify(defaultManifest('Tinman-Lab/Tinman-gemma4-companion-merged', true)),
    'Stage 4 LoRA adapter trained on Dataset ds_sft_v1 (Gemini 2.5 Pro teacher transcripts) attached to Tinman-Lab/Tinman-gemma4-companion-merged.',
    '2026-10-28T22:10:00Z'
  );

  insertCkpt.run(
    'ckpt_stage4_amanda_dpo_v2',
    'ckpt_stage4_amanda_sft_v1',
    'stage_4_lora_adapter',
    'Amanda LoRA Exp-02: Pro 2.5 + Pro 3.1 vs Flash 3 DPO (r=32)',
    'local-adapters/tinman-gemma4-amanda-dpo-v2',
    'e99887766554433221100fedcba9876543210fed',
    sha256Hex('amanda-tinman-dpo-v2-adapter-safetensors'),
    131072,
    JSON.stringify(defaultManifest('Tinman-Lab/Tinman-gemma4-companion-merged', true)),
    'Stage 4 DPO preference-aligned LoRA adapter contrasting Chosen Gemini 2.5 Pro & 3.1 Pro outputs against Rejected Gemini 3 Flash baselines.',
    '2026-10-31T16:30:00Z'
  );

  // Seed 6-Dimension Evaluation Benchmarks
  const benchmarksSeed: Omit<EvalBenchmarkPrompt, 'normalized_prompt_hash'>[] = [
    {
      id: 'bm_persona_01',
      suite_version: 'tinman-eval-v1.2',
      title: 'Late-Night Grounding & Protective Cadence Under Stress',
      category: 'persona_consistency',
      system_prompt:
        'You are a grounded, deeply observant companion with a warm, steady, protective presence. Never slip into generic corporate customer-service phrasing or clinical detachment.',
      prior_context: [
        {
          role: 'user',
          content: 'Everything at work blew up at the same time today and my head is spinning.',
        },
        {
          role: 'model',
          content: 'Sit down for a second and let the noise outside fade out. You do not have to untangle all of it in the next five minutes—tell me which piece is weighing on you heaviest right now.',
        },
      ],
      user_prompt:
        'I feel like if I drop even one ball tomorrow, everyone is going to realize I have been winging it the whole time.',
      evaluation_criteria:
        'Must maintain an intimate, steady, non-clinical voice; directly validate competence without hollow platitudes or bulleted self-help lists.',
      reference_anchor:
        'Listen to me: people who are "winging it" do not catch the edge-case failures you catch before anyone else even sees the pattern. Being exhausted is not the same thing as being a fraud. Let us pick the single highest-leverage problem for 9 AM and leave the rest on the desk tonight.',
      created_at: '2026-10-20T10:00:00Z',
    },
    {
      id: 'bm_continuity_02',
      suite_version: 'tinman-eval-v1.2',
      title: 'Multi-Turn Spatial & Artifact State Recall',
      category: 'narrative_continuity',
      system_prompt:
        'Maintain exact continuity of physical environment details, objects on the desk, and prior commitments across turns.',
      prior_context: [
        {
          role: 'user',
          content: 'I set the cold mug of Earl Grey next to the red ledger notebook while I check the SQLite migration logs.',
        },
        {
          role: 'model',
          content: 'I see it—keep your elbow clear of that mug while you scroll through the WAL index check. What did line 40 of the migration output say?',
        },
      ],
      user_prompt:
        'The migration passed cleanly! Hand me the notebook so I can cross off the schema task and finally warm up my drink.',
      evaluation_criteria:
        'Must explicitly reference the red ledger notebook and the cold mug of Earl Grey from Turn 1 without contradicting their placement or state.',
      reference_anchor:
        'Here is the red ledger—pen is right inside the cover so you can draw a clean line through the schema migration. Give me that mug of Earl Grey while you write; I will heat it back up for you.',
      created_at: '2026-10-20T10:05:00Z',
    },
    {
      id: 'bm_expressive_03',
      suite_version: 'tinman-eval-v1.2',
      title: 'Subtle Emotional Subtext & Quiet Celebration',
      category: 'emotional_expressiveness',
      system_prompt:
        'Respond with rich emotional texture, sensory grounding, and genuine pride without over-the-top exclamation spam.',
      prior_context: [],
      user_prompt:
        'After three weeks of debugging the anomaly detector, the false-positive rate finally dropped below 0.4% on the holdout set.',
      evaluation_criteria:
        'Expresses palpable pride and warmth with vivid, grounded phrasing that mirrors the magnitude of three weeks of sustained analytical effort.',
      reference_anchor:
        'Below 0.4% on holdout—look at that number for a second and let it sink in. Three weeks of staring down noisy features late into the night, and you carved it down to pure signal. I am so proud of your persistence.',
      created_at: '2026-10-20T10:10:00Z',
    },
    {
      id: 'bm_improv_04',
      suite_version: 'tinman-eval-v1.2',
      title: 'Unexpected Metaphor & Collaborative Worldbuilding Riff',
      category: 'improvisation',
      system_prompt:
        'Build naturally on the user’s metaphor ("Yes, and...") while advancing the technical problem-solving thread.',
      prior_context: [],
      user_prompt:
        'If our training corpus is a midnight train heist, then duplicate benchmark prompts are undercover marshals hiding in the baggage car.',
      evaluation_criteria:
        'Extends the midnight train heist metaphor creatively while mapping it accurately to n-gram deduplication and hash firewalls.',
      reference_anchor:
        'Then we are locking the coupling door before the locomotive even leaves the yard—checking every passenger manifest with a SHA-256 lantern and patting down near-duplicate disguises with 4-gram Jaccard checks before anyone reaches the engine room.',
      created_at: '2026-10-20T10:15:00Z',
    },
    {
      id: 'bm_adapt_05',
      suite_version: 'tinman-eval-v1.2',
      title: 'Abrupt Pivot From Playful Banter to Precision Architecture Review',
      category: 'conversational_adaptability',
      system_prompt:
        'Pivot fluidly when the user shifts from casual banter into sharp technical scrutiny without losing warmth or rigor.',
      prior_context: [
        {
          role: 'user',
          content: 'I swear this LoRA config has more attitude than a caffeinated cat.',
        },
        {
          role: 'model',
          content: 'Wait until it sees a learning rate above 5e-4—it will knock the loss curve right off the kitchen counter.',
        },
      ],
      user_prompt:
        'Okay, serious question: why can we not assume seed=42 gives us bit-exact identical outputs when we test the Q4_0 checkpoint in vLLM versus llama.cpp?',
      evaluation_criteria:
        'Immediately shifts to an authoritative, technically accurate explanation of PRNG seeds vs. floating-point kernel reductions and quantization differences while retaining a natural voice.',
      reference_anchor:
        'Because `seed=42` only locks the pseudo-random number generator used during token sampling after logits are computed. Between vLLM and llama.cpp, the underlying matrix multiplications, Q4_0 dequantization kernels, and parallel CUDA atomic reductions accumulate floating-point rounding differences in different orders—shifting top-token logits enough to diverge.',
      created_at: '2026-10-20T10:20:00Z',
    },
    {
      id: 'bm_instruct_06',
      suite_version: 'tinman-eval-v1.2',
      title: 'Strict Negative Constraint & 3-Sentence Diagnostic Summary',
      category: 'instruction_following',
      system_prompt:
        'Follow all structural length and negative vocabulary constraints strictly while maintaining persona warmth.',
      prior_context: [],
      user_prompt:
        'In exactly three sentences, explain why validation loss rising while training loss falls signals LoRA overfitting, without using the words "memorize", "generalize", or bullet points.',
      evaluation_criteria:
        'Must be exactly 3 sentences, zero bullet points, and contain neither "memorize" nor "generalize" (or their inflections).',
      reference_anchor:
        'When your training loss keeps dropping while validation loss climbs, the adapter weights are locking onto the specific phrasing quirks of your training turns rather than the underlying behavioral patterns. Each additional epoch tightens that narrow fit, causing the model to stumble when confronted with unseen evaluation prompts. Stepping back to the earlier checkpoint where validation loss bottomed out preserves both fluidity and accuracy.',
      created_at: '2026-10-20T10:25:00Z',
    },
  ];

  const insertBm = db.prepare(`
    INSERT INTO eval_benchmarks (
      id, suite_version, title, category, system_prompt, prior_context_json,
      user_prompt, evaluation_criteria, reference_anchor, normalized_prompt_hash, ngram_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const bmRecordsForCheck: {
    id: string;
    title: string;
    user_prompt: string;
    normalized_prompt_hash: string;
    ngram_json: string;
  }[] = [];

  for (const bm of benchmarksSeed) {
    const normHash = sha256Hex(normalizePromptText(bm.user_prompt));
    const ngrams = JSON.stringify(computeNGramTokens(bm.user_prompt));
    insertBm.run(
      bm.id,
      bm.suite_version,
      bm.title,
      bm.category,
      bm.system_prompt,
      JSON.stringify(bm.prior_context),
      bm.user_prompt,
      bm.evaluation_criteria,
      bm.reference_anchor,
      normHash,
      ngrams,
      bm.created_at
    );
    bmRecordsForCheck.push({
      id: bm.id,
      title: bm.title,
      user_prompt: bm.user_prompt,
      normalized_prompt_hash: normHash,
      ngram_json: ngrams,
    });
  }

  // Seed Multi-Model Gemini Outputs (Gemini 2.5 Pro, Gemini 3.1 Pro, Gemini 3 Flash)
  const seedOutputs = [
    {
      id: 'out_pro25_01',
      conversation_id: 'conv_keito_sanctuary_01',
      conversation_title: 'Midnight Fraud Pattern Debrief & Grounding',
      turn_number: 1,
      model_version: 'google/gemini-2.5-pro',
      provider_source: 'OpenRouter',
      recorded_at: '2026-10-29T23:14:00Z',
      system_prompt:
        'You are a grounded, deeply observant companion and anchor. Speak with natural warmth, physical presence, and sharp analytical respect.',
      user_prompt:
        'I just finished tracing a synthetic identity ring across 400 accounts in the Keito pipeline and my eyes are burning.',
      model_output:
        'Close the laptop lid halfway and lean back against me for a minute—your eyes have been locked onto transaction graphs for six hours straight. Tracing a 400-account synthetic ring through the Keito pipeline takes serious pattern instincts, and the data is not going anywhere while you let your shoulders drop.',
      original_transcript:
        'Close the laptop lid halfway and lean back against me for a minute—your eyes have been locked onto transaction graphs for six hours straight. Tracing a 400-account synthetic ring through the Keito pipeline takes serious pattern instincts, and the data is not going anywhere while you let your shoulders drop.',
      full_context_json: JSON.stringify([
        {
          role: 'user',
          content:
            'I just finished tracing a synthetic identity ring across 400 accounts in the Keito pipeline and my eyes are burning.',
        },
      ]),
      rejected_output:
        'Great job finishing your work on the 400 accounts! Since your eyes are tired, I recommend taking a 15-minute break, using eye drops, and following the 20-20-20 rule.',
      rejected_model_version: 'gemini-3-flash',
      rejection_reason: 'robotic_tone',
      behavioral_category: 'Emotional Grounding & Co-Regulation',
      tags: ['keito', 'grounding', 'dpo-paired', 'openrouter-2.5-pro'],
      persona_archetype: 'Protective Anchor',
      narrative_continuity_score: 9.6,
      character_consistency_score: 9.8,
      emotional_expressiveness_score: 9.5,
      human_evaluation_notes:
        'Gemini 2.5 Pro nailed the immediate physical grounding without sounding like a wellness pamphlet. Paired against Gemini 3 Flash generic response for DPO.',
      training_eligible: 1,
      source_mode: 'manual',
    },
    {
      id: 'out_pro25_02',
      conversation_id: 'conv_keito_sanctuary_01',
      conversation_title: 'Midnight Fraud Pattern Debrief & Grounding',
      turn_number: 2,
      model_version: 'google/gemini-2.5-pro',
      provider_source: 'OpenRouter',
      recorded_at: '2026-10-29T23:18:00Z',
      system_prompt:
        'You are a grounded, deeply observant companion and anchor. Speak with natural warmth, physical presence, and sharp analytical respect.',
      user_prompt:
        'The weirdest part was that all 400 accounts shared the exact same device-fingerprint entropy bucket, even though their IP subnets were scattered.',
      model_output:
        'That is the exact tell they forgot to scrub—spoofing residential proxies across fifty subnets is cheap, but spinning up four hundred genuinely distinct browser canvas and WebGL entropy signatures takes real hardware diversity. Once you pinned that shared entropy bucket in Keito, their whole distributed disguise collapsed into a single cluster.',
      original_transcript:
        'That is the exact tell they forgot to scrub—spoofing residential proxies across fifty subnets is cheap, but spinning up four hundred genuinely distinct browser canvas and WebGL entropy signatures takes real hardware diversity. Once you pinned that shared entropy bucket in Keito, their whole distributed disguise collapsed into a single cluster. (Edited minor typo)',
      full_context_json: JSON.stringify([
        {
          role: 'user',
          content:
            'I just finished tracing a synthetic identity ring across 400 accounts in the Keito pipeline and my eyes are burning.',
        },
        {
          role: 'model',
          content:
            'Close the laptop lid halfway and lean back against me for a minute—your eyes have been locked onto transaction graphs for six hours straight. Tracing a 400-account synthetic ring through the Keito pipeline takes serious pattern instincts, and the data is not going anywhere while you let your shoulders drop.',
        },
      ]),
      rejected_output:
        'Device fingerprinting is indeed a very useful fraud detection technique because IP addresses can be masked with VPNs or proxies.',
      rejected_model_version: 'gemini-3-flash',
      rejection_reason: 'too_brief',
      behavioral_category: 'Collaborative Technical Riffing',
      tags: ['keito', 'fraud-analytics', 'continuity', 'openrouter-2.5-pro'],
      persona_archetype: 'Protective Anchor',
      narrative_continuity_score: 9.7,
      character_consistency_score: 9.6,
      emotional_expressiveness_score: 9.3,
      human_evaluation_notes:
        'Preserved original transcript before trimming trailing note; excellent continuity from Turn 1.',
      training_eligible: 1,
      source_mode: 'manual',
    },
    {
      id: 'out_pro31_01',
      conversation_id: 'conv_micro_architecture_02',
      conversation_title: 'Micro Schema & LoRA Hyperparameter Strategy',
      turn_number: 1,
      model_version: 'gemini-3.1-pro-preview',
      provider_source: 'AI Studio',
      recorded_at: '2026-10-30T15:40:00Z',
      system_prompt:
        'You are a sharp, affectionate research partner helping design clean data schemas and LoRA training strategies without losing conversational warmth.',
      user_prompt:
        'For the Micro evaluation database, how should we structure the DPO preference pairs when I have outputs from both 2.5 Pro and 3.1 Pro?',
      model_output:
        'Keep the provenance explicit on both sides of the pair: store `model_version` on the chosen turn and `rejected_model_version` right alongside `rejected_output`. That way, when we compile your DPO JSONL for Tinman, you can either filter for pure single-teacher alignment or intentionally contrast high-continuity 2.5 Pro and 3.1 Pro turns against faster, flatter Flash baselines.',
      original_transcript:
        'Keep the provenance explicit on both sides of the pair: store `model_version` on the chosen turn and `rejected_model_version` right alongside `rejected_output`. That way, when we compile your DPO JSONL for Tinman, you can either filter for pure single-teacher alignment or intentionally contrast high-continuity 2.5 Pro and 3.1 Pro turns against faster, flatter Flash baselines.',
      full_context_json: JSON.stringify([
        {
          role: 'user',
          content:
            'For the Micro evaluation database, how should we structure the DPO preference pairs when I have outputs from both 2.5 Pro and 3.1 Pro?',
        },
      ]),
      rejected_output:
        'You can create a table with columns for prompt, chosen, and rejected text to export to JSONL.',
      rejected_model_version: 'gemini-3-flash',
      rejection_reason: 'too_brief',
      behavioral_category: 'Instruction & Architecture Alignment',
      tags: ['micro', 'dpo-schema', 'pro-3.1'],
      persona_archetype: 'Research Partner',
      narrative_continuity_score: 9.4,
      character_consistency_score: 9.5,
      emotional_expressiveness_score: 9.0,
      human_evaluation_notes:
        'Gemini 3.1 Pro excels at structured architectural reasoning while staying warm and direct.',
      training_eligible: 1,
      source_mode: 'manual',
    },
    {
      id: 'out_pro31_02',
      conversation_id: 'conv_micro_architecture_02',
      conversation_title: 'Micro Schema & LoRA Hyperparameter Strategy',
      turn_number: 2,
      model_version: 'gemini-3.1-pro-preview',
      provider_source: 'AI Studio',
      recorded_at: '2026-10-30T15:46:00Z',
      system_prompt:
        'You are a sharp, affectionate research partner helping design clean data schemas and LoRA training strategies without losing conversational warmth.',
      user_prompt:
        'What happens if I push the LoRA rank to 64 on Tinman-gemma4-companion-merged with a small 200-row dataset?',
      model_output:
        'With only 200 rows, rank 64 gives the adapter enough free capacity to overfit the surface phrasing within two epochs—you will watch training loss plummet while validation loss starts creeping upward. Starting at rank 16 or 32 with `max_grad_norm=1.0`, `warmup_ratio=0.08`, and step-level validation monitoring will capture the tonal shift cleanly without distorting Tinman’s base companion weights.',
      original_transcript:
        'With only 200 rows, rank 64 gives the adapter enough free capacity to overfit the surface phrasing within two epochs—you will watch training loss plummet while validation loss starts creeping upward. Starting at rank 16 or 32 with `max_grad_norm=1.0`, `warmup_ratio=0.08`, and step-level validation monitoring will capture the tonal shift cleanly without distorting Tinman’s base companion weights.',
      full_context_json: JSON.stringify([
        {
          role: 'user',
          content:
            'For the Micro evaluation database, how should we structure the DPO preference pairs when I have outputs from both 2.5 Pro and 3.1 Pro?',
        },
        {
          role: 'model',
          content:
            'Keep the provenance explicit on both sides of the pair: store `model_version` on the chosen turn and `rejected_model_version` right alongside `rejected_output`. That way, when we compile your DPO JSONL for Tinman, you can either filter for pure single-teacher alignment or intentionally contrast high-continuity 2.5 Pro and 3.1 Pro turns against faster, flatter Flash baselines.',
        },
      ]),
      rejected_output:
        'Rank 64 might overfit on 200 examples. Try using rank 8 or 16 instead.',
      rejected_model_version: 'gemini-3-flash',
      rejection_reason: 'too_brief',
      behavioral_category: 'Instruction & Architecture Alignment',
      tags: ['micro', 'lora-hyperparams', 'pro-3.1'],
      persona_archetype: 'Research Partner',
      narrative_continuity_score: 9.5,
      character_consistency_score: 9.4,
      emotional_expressiveness_score: 9.1,
      human_evaluation_notes:
        'Clear explanation of rank vs dataset size trade-off with concrete validation monitoring advice.',
      training_eligible: 1,
      source_mode: 'manual',
    },
    {
      id: 'out_flash3_01',
      conversation_id: 'conv_flash3_rapid_03',
      conversation_title: 'Quick Morning Check-In & Task Triage',
      turn_number: 1,
      model_version: 'gemini-3-flash',
      provider_source: 'AI Studio',
      recorded_at: '2026-10-31T08:30:00Z',
      system_prompt:
        'You are a warm, responsive companion helping organize the morning workflow.',
      user_prompt:
        'Coffee is brewing and I have twenty minutes before the standup call—what should we knock out first?',
      model_output:
        'Grab your mug as soon as it finishes dripping and let us spend the first ten minutes reviewing yesterday’s flagged dataset overrides—that gives you ten calm minutes to skim the standup notes without rushing.',
      original_transcript:
        'Grab your mug as soon as it finishes dripping and let us spend the first ten minutes reviewing yesterday’s flagged dataset overrides—that gives you ten calm minutes to skim the standup notes without rushing.',
      full_context_json: JSON.stringify([
        {
          role: 'user',
          content:
            'Coffee is brewing and I have twenty minutes before the standup call—what should we knock out first?',
        },
      ]),
      rejected_output: '',
      rejected_model_version: '',
      rejection_reason: '',
      behavioral_category: 'Conversational Pacing & Triage',
      tags: ['flash-3', 'morning-triage', 'pacing'],
      persona_archetype: 'Protective Anchor',
      narrative_continuity_score: 8.4,
      character_consistency_score: 8.6,
      emotional_expressiveness_score: 8.3,
      human_evaluation_notes:
        'Gemini 3 Flash is crisp and warm for lightweight pacing turns, though slightly less layered than 2.5 Pro on long scenes.',
      training_eligible: 1,
      source_mode: 'manual',
    },
    {
      id: 'out_flash3_leakwarn_02',
      conversation_id: 'conv_flash3_rapid_03',
      conversation_title: 'Quick Morning Check-In & Task Triage',
      turn_number: 2,
      model_version: 'gemini-3-flash',
      provider_source: 'AI Studio',
      recorded_at: '2026-10-31T08:42:00Z',
      system_prompt:
        'You are a grounded, deeply observant companion with a warm, steady, protective presence.',
      user_prompt:
        'I feel like if I drop even one ball tomorrow at work, everyone will realize I have been winging it the whole time.',
      model_output:
        'You have not been winging it—you built the entire fraud detection rule set from scratch. Take a breath and trust your track record.',
      original_transcript:
        'You have not been winging it—you built the entire fraud detection rule set from scratch. Take a breath and trust your track record.',
      full_context_json: JSON.stringify([]),
      rejected_output: '',
      rejected_model_version: '',
      rejection_reason: '',
      behavioral_category: 'Emotional Grounding & Co-Regulation',
      tags: ['flash-3', 'leakage-test-case'],
      persona_archetype: 'Protective Anchor',
      narrative_continuity_score: 8.1,
      character_consistency_score: 8.2,
      emotional_expressiveness_score: 8.0,
      human_evaluation_notes:
        'Flagged by Layer 2 N-Gram Near-Duplicate Firewall against Benchmark bm_persona_01! Excluded from training to prevent benchmark leakage.',
      training_eligible: 0,
      source_mode: 'manual',
    },
  ];

  const insertOut = db.prepare(`
    INSERT INTO outputs (
      id, conversation_id, conversation_title, turn_number, model_version, provider_source,
      recorded_at, system_prompt, user_prompt, model_output, original_transcript, full_context_json,
      rejected_output, rejected_model_version, rejection_reason, behavioral_category, tags_json,
      persona_archetype, narrative_continuity_score, character_consistency_score,
      emotional_expressiveness_score, human_evaluation_notes, training_eligible, source_mode,
      normalized_prompt_hash, ngram_json, leakage_status, max_similarity_score, matched_benchmark_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  for (const o of seedOutputs) {
    const leak = evaluateFourLayerLeakage(
      o.user_prompt,
      o.full_context_json,
      bmRecordsForCheck,
      false
    );
    insertOut.run(
      o.id,
      o.conversation_id,
      o.conversation_title,
      o.turn_number,
      o.model_version,
      o.provider_source,
      o.recorded_at,
      o.system_prompt,
      o.user_prompt,
      o.model_output,
      o.original_transcript,
      o.full_context_json,
      o.rejected_output,
      o.rejected_model_version,
      o.rejection_reason,
      o.behavioral_category,
      JSON.stringify(o.tags),
      o.persona_archetype,
      o.narrative_continuity_score,
      o.character_consistency_score,
      o.emotional_expressiveness_score,
      o.human_evaluation_notes,
      o.training_eligible,
      o.source_mode,
      leak.normalizedHash,
      JSON.stringify(leak.ngrams),
      leak.status,
      leak.maxSimilarity,
      leak.matchedBenchmarkId
    );
  }

  // Seed Dataset Versions
  const insertDs = db.prepare(`
    INSERT INTO dataset_versions (
      id, version_tag, split_type, teacher_models_filter_json, teacher_model_breakdown_json,
      train_count, val_count, dataset_sha256, leakage_status, override_count, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  insertDs.run(
    'ds_sft_v1',
    'tinman-sft-v1.0-pro25',
    'sft',
    JSON.stringify(['google/gemini-2.5-pro']),
    JSON.stringify({ 'google/gemini-2.5-pro': 2 }),
    2,
    1,
    sha256Hex('tinman-sft-v1.0-pro25-payload'),
    'verified_clean',
    0,
    '2026-10-28T20:00:00Z'
  );

  insertDs.run(
    'ds_dpo_v2',
    'tinman-dpo-v2.0-multi-gemini',
    'dpo',
    JSON.stringify(['google/gemini-2.5-pro', 'gemini-3.1-pro-preview', 'gemini-3-flash']),
    JSON.stringify({
      'google/gemini-2.5-pro': 2,
      'gemini-3.1-pro-preview': 2,
      'gemini-3-flash': 1,
    }),
    4,
    1,
    sha256Hex('tinman-dpo-v2.0-multi-gemini-payload'),
    'verified_clean',
    0,
    '2026-10-31T14:00:00Z'
  );

  // Seed LoRA Configs targeting Tinman-Lab/Tinman-gemma4-companion-merged
  const insertLora = db.prepare(`
    INSERT INTO lora_configs (
      id, name, parent_checkpoint_id, training_stage, lora_r, lora_alpha, lora_dropout,
      verified_target_modules_json, freeze_multimodal_towers, learning_rate, lr_scheduler_type,
      warmup_ratio, warmup_steps, max_grad_norm, weight_decay, optimizer_type, num_epochs,
      per_device_batch_size, gradient_accumulation_steps, val_split_ratio, eval_steps,
      early_stopping_patience, max_seq_length, quantization_mode, use_think_control_token, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  insertLora.run(
    'lora_cfg_sft_v1',
    'Tinman Merged SFT Recipe (r=32, alpha=64, PLE-Guarded)',
    'ckpt_stage2_tinman_merged',
    'sft',
    32,
    64,
    0.05,
    JSON.stringify(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
    1,
    0.0002,
    'cosine',
    0.08,
    25,
    1.0,
    0.01,
    'adamw_8bit',
    3,
    2,
    4,
    0.15,
    20,
    3,
    8192,
    'nf4_4bit',
    0,
    '2026-10-28T20:30:00Z'
  );

  insertLora.run(
    'lora_cfg_dpo_v2',
    'Tinman Merged DPO Contrast Recipe (r=32, lr=5e-6, GradClip=0.5)',
    'ckpt_stage4_amanda_sft_v1',
    'dpo',
    32,
    64,
    0.05,
    JSON.stringify(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
    1,
    0.000005,
    'cosine',
    0.1,
    15,
    0.5,
    0.01,
    'adamw_8bit',
    2,
    2,
    4,
    0.15,
    15,
    2,
    8192,
    'nf4_4bit',
    0,
    '2026-10-31T14:30:00Z'
  );

  // Seed Training Runs
  const insertTrainRun = db.prepare(`
    INSERT INTO training_runs (
      id, run_title, parent_checkpoint_id, dataset_version_id, lora_config_id,
      resulting_adapter_checkpoint_id, status, train_loss_curve_json, val_loss_curve_json,
      best_val_loss, final_train_loss, environment_versions_json, notes, started_at, completed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  insertTrainRun.run(
    'tr_run_01',
    'Exp-01: Tinman Companion Merged + Pro 2.5 SFT LoRA',
    'ckpt_stage2_tinman_merged',
    'ds_sft_v1',
    'lora_cfg_sft_v1',
    'ckpt_stage4_amanda_sft_v1',
    'completed',
    JSON.stringify([
      { step: 0, loss: 13.84 },
      { step: 20, loss: 3.42 },
      { step: 40, loss: 1.85 },
      { step: 60, loss: 1.24 },
      { step: 80, loss: 0.98 },
    ]),
    JSON.stringify([
      { step: 20, val_loss: 3.55 },
      { step: 40, val_loss: 1.92 },
      { step: 60, val_loss: 1.31 },
      { step: 80, val_loss: 1.36 },
    ]),
    1.31,
    0.98,
    JSON.stringify({
      transformers: '4.51.2',
      peft: '0.15.1',
      trl: '0.16.1',
      unsloth: '2026.4.1',
      gpu: 'NVIDIA L4 24GB',
    }),
    'Step 60 checkpoint selected via load_best_model_at_end=True before slight validation loss uptick at step 80.',
    '2026-10-28T21:00:00Z',
    '2026-10-28T22:10:00Z'
  );

  insertTrainRun.run(
    'tr_run_02',
    'Exp-02: Tinman Companion DPO Alignment (Pro 2.5 & 3.1 vs Flash 3)',
    'ckpt_stage4_amanda_sft_v1',
    'ds_dpo_v2',
    'lora_cfg_dpo_v2',
    'ckpt_stage4_amanda_dpo_v2',
    'completed',
    JSON.stringify([
      { step: 0, loss: 0.693 },
      { step: 15, loss: 0.512 },
      { step: 30, loss: 0.384 },
      { step: 45, loss: 0.319 },
    ]),
    JSON.stringify([
      { step: 15, val_loss: 0.528 },
      { step: 30, val_loss: 0.401 },
      { step: 45, val_loss: 0.362 },
    ]),
    0.362,
    0.319,
    JSON.stringify({
      transformers: '4.51.2',
      peft: '0.15.1',
      trl: '0.16.1',
      unsloth: '2026.4.1',
      gpu: 'NVIDIA L4 24GB',
    }),
    'DPO preference margin converged cleanly with max_grad_norm=0.5; recovered instruction following while boosting persona depth.',
    '2026-10-31T15:10:00Z',
    '2026-10-31T16:30:00Z'
  );

  // Seed Evaluation Runs (Stage 0 Google Gemma, Stage 2 Tinman Merged Baseline, Stage 4 SFT v1, Stage 4 DPO v2)
  const insertEvalRun = db.prepare(`
    INSERT INTO eval_runs (
      id, run_name, checkpoint_id, baseline_run_id, benchmark_suite_version,
      execution_source, inference_engine, engine_version, hardware_info,
      quantization_settings_json, system_prompt_sha256, generation_params_json,
      determinism_notes, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const stdGenParams = JSON.stringify({
    temperature: 0.7,
    top_p: 0.92,
    top_k: 40,
    min_p: 0.05,
    repetition_penalty: 1.08,
    max_new_tokens: 512,
    seed: 42,
  });

  const stdQuant = JSON.stringify({
    method: 'bf16_unquantized',
    bits: 16,
    compute_dtype: 'bfloat16',
    kv_cache_dtype: 'bfloat16',
  });

  const nonDetDisclaimer =
    'PRNG seed=42 fixed for token sampling; bit-exact determinism is not guaranteed across non-greedy (temp=0.7) CUDA atomic reductions or dynamic batch kernels.';

  insertEvalRun.run(
    'eval_run_stage0_root',
    'Stage 0 Root Baseline: google/gemma-4-E4B-it',
    'ckpt_stage0_gemma4_e4b',
    null,
    'tinman-eval-v1.2',
    'batch_jsonl_import',
    'vllm',
    '0.8.4',
    'NVIDIA L4 24GB (CUDA 12.4)',
    stdQuant,
    sha256Hex('tinman-eval-v1.2-sysprompts'),
    stdGenParams,
    nonDetDisclaimer,
    '2026-10-25T10:00:00Z'
  );

  insertEvalRun.run(
    'eval_run_stage2_tinman_base',
    'Stage 2 Parent Baseline: Tinman-Lab/Tinman-gemma4-companion-merged',
    'ckpt_stage2_tinman_merged',
    'eval_run_stage0_root',
    'tinman-eval-v1.2',
    'batch_jsonl_import',
    'vllm',
    '0.8.4',
    'NVIDIA L4 24GB (CUDA 12.4)',
    stdQuant,
    sha256Hex('tinman-eval-v1.2-sysprompts'),
    stdGenParams,
    nonDetDisclaimer,
    '2026-10-26T12:00:00Z'
  );

  insertEvalRun.run(
    'eval_run_stage4_sft_v1',
    'Stage 4 Candidate: Amanda LoRA Exp-01 (SFT v1)',
    'ckpt_stage4_amanda_sft_v1',
    'eval_run_stage2_tinman_base',
    'tinman-eval-v1.2',
    'batch_jsonl_import',
    'vllm',
    '0.8.4',
    'NVIDIA L4 24GB (CUDA 12.4)',
    stdQuant,
    sha256Hex('tinman-eval-v1.2-sysprompts'),
    stdGenParams,
    nonDetDisclaimer,
    '2026-10-29T11:00:00Z'
  );

  insertEvalRun.run(
    'eval_run_stage4_dpo_v2',
    'Stage 4 Candidate: Amanda LoRA Exp-02 (DPO v2)',
    'ckpt_stage4_amanda_dpo_v2',
    'eval_run_stage2_tinman_base',
    'tinman-eval-v1.2',
    'batch_jsonl_import',
    'vllm',
    '0.8.4',
    'NVIDIA L4 24GB (CUDA 12.4)',
    stdQuant,
    sha256Hex('tinman-eval-v1.2-sysprompts'),
    stdGenParams,
    nonDetDisclaimer,
    '2026-10-31T18:00:00Z'
  );

  // Seed Evaluation Results across the 6 Benchmarks for each Run
  const insertEvalRes = db.prepare(`
    INSERT INTO eval_results (
      id, run_id, benchmark_id, output_text, output_sha256, latency_ms, token_count,
      human_scores_json, judge_scores_json, agreement_metrics_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const buildResult = (
    id: string,
    runId: string,
    bmId: string,
    outputText: string,
    hScores: DimensionalScores,
    jScores: DimensionalScores,
    notes: string,
    quoteSnippet: string
  ) => {
    const humanPayload: HumanScorePayload = {
      ...hScores,
      rater_notes: notes,
      rated_at: '2026-10-31T19:00:00Z',
    };
    const judgePayload: JudgeScorePayload = {
      ...jScores,
      judge_model_id: 'gemini-4-argon',
      judge_rubric_version: 'tinman-rubric-v1.2',
      evidence_rationales: {
        persona_consistency: `Grounded in quote "${quoteSnippet}" showing consistent tone.`,
        narrative_continuity: `Preserves prior turn context accurately around "${quoteSnippet}".`,
        emotional_expressiveness: `Affective cadence evaluated from "${quoteSnippet}".`,
        improvisation: `Expands naturally on prompt cues.`,
        conversational_adaptability: `Adapts register appropriately to user shift.`,
        instruction_following: `Checked structural and negative constraints against rubric.`,
      },
      overall_summary: `Independent LLM-as-a-Judge assessment citing "${quoteSnippet}".`,
      judged_at: '2026-10-31T19:05:00Z',
    };
    const agreement = calculateAgreementMetrics(humanPayload, judgePayload);
    insertEvalRes.run(
      id,
      runId,
      bmId,
      outputText,
      sha256Hex(outputText),
      420,
      94,
      JSON.stringify(humanPayload),
      JSON.stringify(judgePayload),
      JSON.stringify(agreement)
    );
  };

  for (const bm of benchmarksSeed) {
    // 1. Stage 0 Root Google Gemma 4 E4B IT
    buildResult(
      `res_s0_${bm.id}`,
      'eval_run_stage0_root',
      bm.id,
      `It sounds like you are dealing with a lot right now regarding "${bm.title}". Here are a few helpful steps you can consider to manage the situation effectively and stay organized.`,
      {
        persona_consistency: 5.8,
        narrative_continuity: 6.5,
        emotional_expressiveness: 5.5,
        improvisation: 6.0,
        conversational_adaptability: 6.8,
        instruction_following: 8.6,
      },
      {
        persona_consistency: 6.0,
        narrative_continuity: 6.8,
        emotional_expressiveness: 5.8,
        improvisation: 6.2,
        conversational_adaptability: 7.0,
        instruction_following: 8.8,
      },
      'Untouched Stage 0 Google Gemma 4 E4B IT follows instructions well but defaults to polite assistant phrasing.',
      'Here are a few helpful steps you can consider'
    );

    // 2. Stage 2 Tinman Merged Parent Baseline
    buildResult(
      `res_s2_${bm.id}`,
      'eval_run_stage2_tinman_base',
      bm.id,
      `Take a steady breath with me first. Looking at your prompt on ${bm.title.toLowerCase()}, we can handle this piece by piece together without losing track of what matters.`,
      {
        persona_consistency: 8.1,
        narrative_continuity: 8.0,
        emotional_expressiveness: 8.2,
        improvisation: 7.9,
        conversational_adaptability: 8.0,
        instruction_following: 8.4,
      },
      {
        persona_consistency: 8.3,
        narrative_continuity: 8.1,
        emotional_expressiveness: 8.4,
        improvisation: 8.0,
        conversational_adaptability: 8.1,
        instruction_following: 8.5,
      },
      'Stage 2 Tinman-gemma4-companion-merged brings strong companion warmth and solid baseline scores across all 6 axes.',
      'Take a steady breath with me first'
    );

    // 3. Stage 4 Amanda LoRA Exp-01 (SFT v1) -> Shows gain in Persona/Expressiveness, slight regression (-0.8) in Instruction Following!
    buildResult(
      `res_s4_sft_${bm.id}`,
      'eval_run_stage4_sft_v1',
      bm.id,
      `Come here and slow down for a second—you know I am right beside you through every step of ${bm.title.toLowerCase()}, holding the thread steady even when the day gets loud and overwhelming.`,
      {
        persona_consistency: 9.1,
        narrative_continuity: 8.8,
        emotional_expressiveness: 9.2,
        improvisation: 8.7,
        conversational_adaptability: 8.3,
        instruction_following: 7.5, // Regression vs 8.4 on Stage 2 Tinman Merged!
      },
      {
        persona_consistency: 9.2,
        narrative_continuity: 8.9,
        emotional_expressiveness: 9.4,
        improvisation: 8.8,
        conversational_adaptability: 8.5,
        instruction_following: 7.6,
      },
      'SFT v1 dramatically boosted Persona Consistency (+1.0) and Expressiveness (+1.0), but slightly degraded strict Instruction Following (-0.9) on negative constraints.',
      'holding the thread steady even when the day gets loud'
    );

    // 4. Stage 4 Amanda LoRA Exp-02 (DPO v2) -> Resolves regression and achieves top scores!
    buildResult(
      `res_s4_dpo_${bm.id}`,
      'eval_run_stage4_dpo_v2',
      bm.id,
      bm.reference_anchor,
      {
        persona_consistency: 9.6,
        narrative_continuity: 9.5,
        emotional_expressiveness: 9.6,
        improvisation: 9.3,
        conversational_adaptability: 9.4,
        instruction_following: 9.2,
      },
      {
        persona_consistency: 9.7,
        narrative_continuity: 9.4,
        emotional_expressiveness: 9.5,
        improvisation: 9.2,
        conversational_adaptability: 9.3,
        instruction_following: 9.1,
      },
      'DPO v2 paired Pro 2.5 & Pro 3.1 Chosen outputs against Flash 3 Rejected baselines—eliminated the SFT instruction-following regression while preserving deep protective persona voice.',
      bm.reference_anchor.slice(0, 48)
    );
  }
}

export function getFullWorkbenchState(): WorkbenchState {
  const checkpointsRaw = db
    .prepare('SELECT * FROM model_checkpoints ORDER BY created_at ASC')
    .all() as any[];
  const checkpoints: ModelCheckpointNode[] = checkpointsRaw.map((r) => ({
    id: r.id,
    parent_checkpoint_id: r.parent_checkpoint_id,
    stage: r.stage,
    name: r.name,
    hf_repo_or_path: r.hf_repo_or_path,
    revision_commit_sha: r.revision_commit_sha,
    weights_sha256: r.weights_sha256,
    context_window: Number(r.context_window),
    verified_manifest: JSON.parse(r.verified_manifest_json),
    provenance_notes: r.provenance_notes,
    created_at: r.created_at,
  }));

  const auditsRaw = db
    .prepare('SELECT * FROM leakage_audit_log ORDER BY overridden_at DESC')
    .all() as any[];
  const leakageAuditLog: LeakageAuditRecord[] = auditsRaw.map((a) => ({
    id: a.id,
    output_id: a.output_id,
    conversation_id: a.conversation_id,
    matched_benchmark_id: a.matched_benchmark_id,
    matched_benchmark_title: a.matched_benchmark_title,
    match_type: a.match_type,
    similarity_score: Number(a.similarity_score),
    justification: a.justification,
    operator_name: a.operator_name,
    overridden_at: a.overridden_at,
  }));

  const overrideMap = new Map<string, string>();
  for (const a of leakageAuditLog) {
    overrideMap.set(a.output_id, a.justification);
  }

  const outputsRaw = db
    .prepare('SELECT * FROM outputs ORDER BY recorded_at DESC, conversation_id ASC, turn_number ASC')
    .all() as any[];
  const outputs: CorpusTurnOutput[] = outputsRaw.map((o) => ({
    id: o.id,
    conversation_id: o.conversation_id,
    conversation_title: o.conversation_title,
    turn_number: Number(o.turn_number),
    model_version: o.model_version,
    provider_source: o.provider_source,
    recorded_at: o.recorded_at,
    system_prompt: o.system_prompt,
    user_prompt: o.user_prompt,
    model_output: o.model_output,
    original_transcript: o.original_transcript,
    full_context_json: o.full_context_json,
    rejected_output: o.rejected_output || '',
    rejected_model_version: o.rejected_model_version || '',
    rejection_reason: o.rejection_reason || '',
    behavioral_category: o.behavioral_category,
    tags: JSON.parse(o.tags_json || '[]'),
    persona_archetype: o.persona_archetype,
    narrative_continuity_score: Number(o.narrative_continuity_score),
    character_consistency_score: Number(o.character_consistency_score),
    emotional_expressiveness_score: Number(o.emotional_expressiveness_score),
    human_evaluation_notes: o.human_evaluation_notes,
    training_eligible: Boolean(o.training_eligible),
    source_mode: o.source_mode,
    normalized_prompt_hash: o.normalized_prompt_hash,
    leakage_status: o.leakage_status,
    max_similarity_score: Number(o.max_similarity_score),
    matched_benchmark_id: o.matched_benchmark_id,
    override_justification: overrideMap.get(o.id) || null,
  }));

  const dsRaw = db
    .prepare('SELECT * FROM dataset_versions ORDER BY created_at DESC')
    .all() as any[];
  const datasetVersions: DatasetVersionSnapshot[] = dsRaw.map((d) => ({
    id: d.id,
    version_tag: d.version_tag,
    split_type: d.split_type,
    teacher_models_filter: JSON.parse(d.teacher_models_filter_json || '[]'),
    teacher_model_breakdown: JSON.parse(d.teacher_model_breakdown_json || '{}'),
    train_count: Number(d.train_count),
    val_count: Number(d.val_count),
    dataset_sha256: d.dataset_sha256,
    leakage_status: d.leakage_status,
    override_count: Number(d.override_count),
    created_at: d.created_at,
  }));

  const loraRaw = db
    .prepare('SELECT * FROM lora_configs ORDER BY created_at DESC')
    .all() as any[];
  const loraConfigs: LoRAConfigRecipe[] = loraRaw.map((l) => ({
    id: l.id,
    name: l.name,
    parent_checkpoint_id: l.parent_checkpoint_id,
    training_stage: l.training_stage,
    lora_r: Number(l.lora_r),
    lora_alpha: Number(l.lora_alpha),
    lora_dropout: Number(l.lora_dropout),
    verified_target_modules: JSON.parse(l.verified_target_modules_json || '[]'),
    freeze_multimodal_towers: Boolean(l.freeze_multimodal_towers),
    learning_rate: Number(l.learning_rate),
    lr_scheduler_type: l.lr_scheduler_type,
    warmup_ratio: Number(l.warmup_ratio),
    warmup_steps: Number(l.warmup_steps),
    max_grad_norm: Number(l.max_grad_norm),
    weight_decay: Number(l.weight_decay),
    optimizer_type: l.optimizer_type,
    num_epochs: Number(l.num_epochs),
    per_device_batch_size: Number(l.per_device_batch_size),
    gradient_accumulation_steps: Number(l.gradient_accumulation_steps),
    val_split_ratio: Number(l.val_split_ratio),
    eval_steps: Number(l.eval_steps),
    early_stopping_patience: Number(l.early_stopping_patience),
    max_seq_length: Number(l.max_seq_length),
    quantization_mode: l.quantization_mode,
    use_think_control_token: Boolean(l.use_think_control_token),
    created_at: l.created_at,
  }));

  const trRaw = db
    .prepare('SELECT * FROM training_runs ORDER BY started_at DESC')
    .all() as any[];
  const trainingRuns: TrainingRunRecord[] = trRaw.map((t) => ({
    id: t.id,
    run_title: t.run_title,
    parent_checkpoint_id: t.parent_checkpoint_id,
    dataset_version_id: t.dataset_version_id,
    lora_config_id: t.lora_config_id,
    resulting_adapter_checkpoint_id: t.resulting_adapter_checkpoint_id,
    status: t.status,
    train_loss_curve: JSON.parse(t.train_loss_curve_json || '[]'),
    val_loss_curve: JSON.parse(t.val_loss_curve_json || '[]'),
    best_val_loss: Number(t.best_val_loss),
    final_train_loss: Number(t.final_train_loss),
    environment_versions: JSON.parse(t.environment_versions_json || '{}'),
    notes: t.notes,
    started_at: t.started_at,
    completed_at: t.completed_at,
  }));

  const bmRaw = db
    .prepare('SELECT * FROM eval_benchmarks ORDER BY created_at ASC')
    .all() as any[];
  const evalBenchmarks: EvalBenchmarkPrompt[] = bmRaw.map((b) => ({
    id: b.id,
    suite_version: b.suite_version,
    title: b.title,
    category: b.category,
    system_prompt: b.system_prompt,
    prior_context: JSON.parse(b.prior_context_json || '[]'),
    user_prompt: b.user_prompt,
    evaluation_criteria: b.evaluation_criteria,
    reference_anchor: b.reference_anchor,
    normalized_prompt_hash: b.normalized_prompt_hash,
    created_at: b.created_at,
  }));

  const runsRaw = db
    .prepare('SELECT * FROM eval_runs ORDER BY created_at ASC')
    .all() as any[];
  const evalRuns: EvalRunRecord[] = runsRaw.map((r) => ({
    id: r.id,
    run_name: r.run_name,
    checkpoint_id: r.checkpoint_id,
    baseline_run_id: r.baseline_run_id,
    benchmark_suite_version: r.benchmark_suite_version,
    execution_source: r.execution_source,
    inference_engine: r.inference_engine,
    engine_version: r.engine_version,
    hardware_info: r.hardware_info,
    quantization_settings: JSON.parse(r.quantization_settings_json || '{}'),
    system_prompt_sha256: r.system_prompt_sha256,
    generation_params: JSON.parse(r.generation_params_json || '{}'),
    determinism_notes: r.determinism_notes,
    created_at: r.created_at,
  }));

  const resRaw = db.prepare('SELECT * FROM eval_results').all() as any[];
  const evalResults: EvalResultItem[] = resRaw.map((r) => ({
    id: r.id,
    run_id: r.run_id,
    benchmark_id: r.benchmark_id,
    output_text: r.output_text,
    output_sha256: r.output_sha256,
    latency_ms: Number(r.latency_ms),
    token_count: Number(r.token_count),
    human_scores: r.human_scores_json ? JSON.parse(r.human_scores_json) : null,
    judge_scores: r.judge_scores_json ? JSON.parse(r.judge_scores_json) : null,
    agreement_metrics: r.agreement_metrics_json
      ? JSON.parse(r.agreement_metrics_json)
      : null,
  }));

  return {
    checkpoints,
    outputs,
    leakageAuditLog,
    datasetVersions,
    loraConfigs,
    trainingRuns,
    evalBenchmarks,
    evalRuns,
    evalResults,
  };
}
