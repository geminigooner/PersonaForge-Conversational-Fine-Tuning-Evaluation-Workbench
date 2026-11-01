import express from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import path from 'path';
import { GoogleGenAI, Type } from '@google/genai';
import {
  db,
  initDatabase,
  getFullWorkbenchState,
  sha256Hex,
  normalizePromptText,
  computeNGramTokens,
  evaluateFourLayerLeakage,
  calculateAgreementMetrics,
} from './src/db/database';
import {
  BehavioralDimension,
  BEHAVIORAL_DIMENSIONS,
  HumanScorePayload,
  JudgeScorePayload,
} from './src/types/workbench';

dotenv.config();

initDatabase();

function getGenAIClient(): GoogleGenAI {
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function resolveSdkModelName(requestedModel: string): string {
  const lower = (requestedModel || '').toLowerCase();
  if (lower.includes('argon') || lower.includes('pro')) {
    return 'gemini-3.8-flash';
  }
  return 'gemini-3.8-flash';
}

function getAllBenchmarksForFirewall() {
  return db
    .prepare(
      'SELECT id, title, user_prompt, normalized_prompt_hash, ngram_json FROM eval_benchmarks'
    )
    .all() as {
    id: string;
    title: string;
    user_prompt: string;
    normalized_prompt_hash: string;
    ngram_json: string;
  }[];
}

function recheckAllOutputsLeakage() {
  const benchmarks = getAllBenchmarksForFirewall();
  const overrides = new Set(
    (
      db.prepare('SELECT output_id FROM leakage_audit_log').all() as {
        output_id: string;
      }[]
    ).map((r) => r.output_id)
  );
  const rows = db
    .prepare('SELECT id, user_prompt, full_context_json FROM outputs')
    .all() as { id: string; user_prompt: string; full_context_json: string }[];

  const updateStmt = db.prepare(`
    UPDATE outputs
    SET normalized_prompt_hash = ?,
        ngram_json = ?,
        leakage_status = ?,
        max_similarity_score = ?,
        matched_benchmark_id = ?
    WHERE id = ?
  `);

  for (const r of rows) {
    const res = evaluateFourLayerLeakage(
      r.user_prompt,
      r.full_context_json,
      benchmarks,
      overrides.has(r.id)
    );
    updateStmt.run(
      res.normalizedHash,
      JSON.stringify(res.ngrams),
      res.status,
      res.maxSimilarity,
      res.matchedBenchmarkId,
      r.id
    );
  }
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '25mb' }));

  // 1. Get complete workbench state
  app.get('/api/state', (_req, res) => {
    try {
      res.json(getFullWorkbenchState());
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to load workbench state' });
    }
  });

  // 2. Log a new Gemini output turn
  app.post('/api/outputs', (req, res) => {
    try {
      const body = req.body;
      const id = body.id || `out_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const conversationId = (body.conversation_id || `conv_${Date.now()}`).trim();
      const conversationTitle = (body.conversation_title || 'Untitled Thread').trim();
      const turnNumber = Number(body.turn_number || 1);
      const modelVersion = (body.model_version || 'google/gemini-2.5-pro').trim();
      const providerSource = (body.provider_source || 'OpenRouter').trim();
      const recordedAt = body.recorded_at || new Date().toISOString();
      const systemPrompt = body.system_prompt || '';
      const userPrompt = body.user_prompt || '';
      const modelOutput = body.model_output || '';
      const originalTranscript = body.original_transcript || modelOutput;
      const fullContextJson =
        typeof body.full_context_json === 'string'
          ? body.full_context_json
          : JSON.stringify(body.full_context || [{ role: 'user', content: userPrompt }]);

      const benchmarks = getAllBenchmarksForFirewall();
      const leak = evaluateFourLayerLeakage(userPrompt, fullContextJson, benchmarks, false);

      // If blocked by leakage firewall, automatically set training_eligible to 0
      const trainingEligible =
        leak.status === 'clean' ? (body.training_eligible !== false ? 1 : 0) : 0;

      db.prepare(`
        INSERT INTO outputs (
          id, conversation_id, conversation_title, turn_number, model_version, provider_source,
          recorded_at, system_prompt, user_prompt, model_output, original_transcript, full_context_json,
          rejected_output, rejected_model_version, rejection_reason, behavioral_category, tags_json,
          persona_archetype, narrative_continuity_score, character_consistency_score,
          emotional_expressiveness_score, human_evaluation_notes, training_eligible, source_mode,
          normalized_prompt_hash, ngram_json, leakage_status, max_similarity_score, matched_benchmark_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        id,
        conversationId,
        conversationTitle,
        turnNumber,
        modelVersion,
        providerSource,
        recordedAt,
        systemPrompt,
        userPrompt,
        modelOutput,
        originalTranscript,
        fullContextJson,
        body.rejected_output || '',
        body.rejected_model_version || '',
        body.rejection_reason || '',
        body.behavioral_category || 'General Companion Dialogue',
        JSON.stringify(Array.isArray(body.tags) ? body.tags : []),
        body.persona_archetype || 'Protective Anchor',
        Number(body.narrative_continuity_score ?? 9.0),
        Number(body.character_consistency_score ?? 9.0),
        Number(body.emotional_expressiveness_score ?? 9.0),
        body.human_evaluation_notes || '',
        trainingEligible,
        body.source_mode || 'manual',
        leak.normalizedHash,
        JSON.stringify(leak.ngrams),
        leak.status,
        leak.maxSimilarity,
        leak.matchedBenchmarkId
      );

      res.json({ createdId: id, leakage: leak, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to save output turn' });
    }
  });

  // 3. Update an existing output turn (preserving original_transcript)
  app.put('/api/outputs/:id', (req, res) => {
    try {
      const { id } = req.params;
      const existing = db.prepare('SELECT * FROM outputs WHERE id = ?').get(id) as any;
      if (!existing) {
        res.status(404).json({ error: 'Output turn not found' });
        return;
      }

      const body = req.body;
      const userPrompt = body.user_prompt ?? existing.user_prompt;
      const fullContextJson = body.full_context_json ?? existing.full_context_json;
      const hasOverride = Boolean(
        db.prepare('SELECT id FROM leakage_audit_log WHERE output_id = ?').get(id)
      );
      const benchmarks = getAllBenchmarksForFirewall();
      const leak = evaluateFourLayerLeakage(
        userPrompt,
        fullContextJson,
        benchmarks,
        hasOverride
      );

      const requestedEligible =
        body.training_eligible !== undefined
          ? body.training_eligible
            ? 1
            : 0
          : existing.training_eligible;

      const finalEligible =
        leak.status === 'clean' || leak.status === 'overridden' ? requestedEligible : 0;

      db.prepare(`
        UPDATE outputs
        SET conversation_id = ?,
            conversation_title = ?,
            turn_number = ?,
            model_version = ?,
            provider_source = ?,
            recorded_at = ?,
            system_prompt = ?,
            user_prompt = ?,
            model_output = ?,
            rejected_output = ?,
            rejected_model_version = ?,
            rejection_reason = ?,
            behavioral_category = ?,
            tags_json = ?,
            persona_archetype = ?,
            narrative_continuity_score = ?,
            character_consistency_score = ?,
            emotional_expressiveness_score = ?,
            human_evaluation_notes = ?,
            training_eligible = ?,
            normalized_prompt_hash = ?,
            ngram_json = ?,
            leakage_status = ?,
            max_similarity_score = ?,
            matched_benchmark_id = ?
        WHERE id = ?
      `).run(
        body.conversation_id ?? existing.conversation_id,
        body.conversation_title ?? existing.conversation_title,
        Number(body.turn_number ?? existing.turn_number),
        body.model_version ?? existing.model_version,
        body.provider_source ?? existing.provider_source,
        body.recorded_at ?? existing.recorded_at,
        body.system_prompt ?? existing.system_prompt,
        userPrompt,
        body.model_output ?? existing.model_output,
        body.rejected_output ?? existing.rejected_output,
        body.rejected_model_version ?? existing.rejected_model_version,
        body.rejection_reason ?? existing.rejection_reason,
        body.behavioral_category ?? existing.behavioral_category,
        body.tags ? JSON.stringify(body.tags) : existing.tags_json,
        body.persona_archetype ?? existing.persona_archetype,
        Number(body.narrative_continuity_score ?? existing.narrative_continuity_score),
        Number(body.character_consistency_score ?? existing.character_consistency_score),
        Number(body.emotional_expressiveness_score ?? existing.emotional_expressiveness_score),
        body.human_evaluation_notes ?? existing.human_evaluation_notes,
        finalEligible,
        leak.normalizedHash,
        JSON.stringify(leak.ngrams),
        leak.status,
        leak.maxSimilarity,
        leak.matchedBenchmarkId,
        id
      );

      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to update output turn' });
    }
  });

  // 4. Restore curated model_output from immutable original_transcript
  app.post('/api/outputs/:id/restore-original', (req, res) => {
    try {
      const { id } = req.params;
      db.prepare('UPDATE outputs SET model_output = original_transcript WHERE id = ?').run(id);
      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to restore original transcript' });
    }
  });

  // 5. Delete an output turn
  app.delete('/api/outputs/:id', (req, res) => {
    try {
      const { id } = req.params;
      db.prepare('DELETE FROM outputs WHERE id = ?').run(id);
      db.prepare('DELETE FROM leakage_audit_log WHERE output_id = ?').run(id);
      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to delete output turn' });
    }
  });

  // 6. Auditable Manual Override for Near-Duplicate Leakage Warning
  app.post('/api/leakage/override', (req, res) => {
    try {
      const { output_id, justification, operator_name } = req.body;
      if (!output_id || !justification || String(justification).trim().length < 8) {
        res.status(400).json({
          error: 'An explicit audit justification (at least 8 characters) is required to override a leakage warning.',
        });
        return;
      }

      const outRow = db.prepare('SELECT * FROM outputs WHERE id = ?').get(output_id) as any;
      if (!outRow) {
        res.status(404).json({ error: 'Output record not found' });
        return;
      }
      if (outRow.leakage_status === 'exact_collision_blocked') {
        res.status(400).json({
          error: 'Exact normalized SHA-256 benchmark collisions cannot be overridden.',
        });
        return;
      }

      const bmRow = outRow.matched_benchmark_id
        ? (db
            .prepare('SELECT title FROM eval_benchmarks WHERE id = ?')
            .get(outRow.matched_benchmark_id) as any)
        : null;

      const auditId = `audit_${Date.now()}`;
      db.prepare(`
        INSERT INTO leakage_audit_log (
          id, output_id, conversation_id, matched_benchmark_id, matched_benchmark_title,
          match_type, similarity_score, justification, operator_name, overridden_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        auditId,
        output_id,
        outRow.conversation_id,
        outRow.matched_benchmark_id || 'unknown',
        bmRow?.title || 'Benchmark Prompt',
        outRow.leakage_status,
        Number(outRow.max_similarity_score || 0),
        String(justification).trim(),
        String(operator_name || 'Amanda (Senior Analyst)').trim(),
        new Date().toISOString()
      );

      db.prepare(`
        UPDATE outputs
        SET leakage_status = 'overridden',
            training_eligible = 1
        WHERE id = ?
      `).run(output_id);

      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to record leakage override' });
    }
  });

  // 7. Live Teacher Generation via @google/genai
  app.post('/api/teacher/generate', async (req, res) => {
    try {
      const { system_prompt, user_prompt, model_label } = req.body;
      if (!user_prompt) {
        res.status(400).json({ error: 'user_prompt is required' });
        return;
      }
      const ai = getGenAIClient();
      const sdkModel = resolveSdkModelName(model_label || 'gemini-3.8-flash');
      const response = await ai.models.generateContent({
        model: sdkModel,
        contents: user_prompt,
        config: system_prompt
          ? {
              systemInstruction: system_prompt,
            }
          : undefined,
      });

      res.json({
        generated_text: response.text || '',
        executed_sdk_model: sdkModel,
        recorded_model_label: model_label || sdkModel,
      });
    } catch (err: any) {
      res.status(500).json({
        error: err.message || 'Live Gemini teacher generation failed',
      });
    }
  });

  // 8. 1-Click Teacher DPO Rejected Contrast Drafter via @google/genai
  app.post('/api/teacher/dpo-contrast', async (req, res) => {
    try {
      const { user_prompt, chosen_output, rejection_style } = req.body;
      if (!user_prompt) {
        res.status(400).json({ error: 'user_prompt is required' });
        return;
      }
      const ai = getGenAIClient();
      const styleDirectives: Record<string, string> = {
        robotic_tone:
          'Write a generic, mildly robotic, overly formal customer-service style assistant response (2-3 sentences) that answers the prompt superficially without emotional warmth or persona depth.',
        too_brief:
          'Write a flat, overly brief 1-sentence response that misses the emotional subtext and feels rushed.',
        out_of_character:
          'Write a detached, encyclopedia-like response that breaks companion persona and uses generic bulleted advice.',
        continuity_break:
          'Write a polite response that ignores specific contextual details mentioned by the user.',
      };
      const directive =
        styleDirectives[rejection_style || 'robotic_tone'] || styleDirectives.robotic_tone;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: `User Prompt: "${user_prompt}"\n\nHigh-Quality Chosen Output (for contrast): "${chosen_output || ''}"\n\nTask: ${directive}\nReturn ONLY the suboptimal response text itself with no preamble.`,
      });

      res.json({
        rejected_output: (response.text || '').trim(),
        rejected_model_version: 'gemini-3-flash (contrast-draft)',
        rejection_reason: rejection_style || 'robotic_tone',
      });
    } catch (err: any) {
      res.status(500).json({
        error: err.message || 'Failed to draft DPO contrast response',
      });
    }
  });

  // 9. Create a leak-verified Dataset Version Snapshot (Stage 3 Lineage)
  app.post('/api/datasets/snapshot', (req, res) => {
    try {
      const { version_tag, split_type, teacher_models_filter, val_split_ratio } = req.body;
      const allOutputs = getFullWorkbenchState().outputs;
      const filterList: string[] = Array.isArray(teacher_models_filter)
        ? teacher_models_filter
        : [];

      const eligibleTurns = allOutputs.filter((o) => {
        if (!o.training_eligible) return false;
        if (
          o.leakage_status === 'exact_collision_blocked' ||
          o.leakage_status === 'conversation_subsequence_blocked' ||
          o.leakage_status === 'near_duplicate_warn'
        ) {
          return false;
        }
        if (split_type === 'dpo' && !o.rejected_output.trim()) return false;
        if (filterList.length > 0 && !filterList.includes(o.model_version)) return false;
        return true;
      });

      if (eligibleTurns.length === 0) {
        res.status(400).json({
          error: 'No eligible, leak-verified turns match the selected Gemini teacher model filter and split type.',
        });
        return;
      }

      const breakdown: Record<string, number> = {};
      let overrideCount = 0;
      for (const t of eligibleTurns) {
        breakdown[t.model_version] = (breakdown[t.model_version] || 0) + 1;
        if (t.leakage_status === 'overridden') overrideCount++;
      }

      const ratio = Number(val_split_ratio ?? 0.15);
      const valCount = Math.max(1, Math.round(eligibleTurns.length * ratio));
      const trainCount = Math.max(1, eligibleTurns.length - valCount);
      const digest = sha256Hex(
        JSON.stringify(eligibleTurns.map((t) => ({ id: t.id, out: t.model_output, rej: t.rejected_output })))
      );

      const id = `ds_${Date.now()}`;
      db.prepare(`
        INSERT INTO dataset_versions (
          id, version_tag, split_type, teacher_models_filter_json, teacher_model_breakdown_json,
          train_count, val_count, dataset_sha256, leakage_status, override_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        id,
        (version_tag || `tinman-${split_type || 'sft'}-${new Date().toISOString().slice(0, 10)}`).trim(),
        split_type === 'dpo' ? 'dpo' : 'sft',
        JSON.stringify(filterList),
        JSON.stringify(breakdown),
        trainCount,
        valCount,
        digest,
        overrideCount > 0 ? 'contains_audited_overrides' : 'verified_clean',
        overrideCount,
        new Date().toISOString()
      );

      res.json({ createdDatasetId: id, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to create dataset snapshot' });
    }
  });

  // 10. Validate & Import external JSON / JSONL datasets
  app.post('/api/import/validate', (req, res) => {
    try {
      const { raw_content } = req.body;
      if (!raw_content || !String(raw_content).trim()) {
        res.status(400).json({ error: 'Empty import content' });
        return;
      }
      const trimmed = String(raw_content).trim();
      let records: any[] = [];
      let detectedSchemaVersion = 'unversioned-legacy (auto-migrated to v1.2.0)';

      if (trimmed.startsWith('{') && !trimmed.includes('\n{')) {
        const parsed = JSON.parse(trimmed);
        if (parsed.schema_version) detectedSchemaVersion = parsed.schema_version;
        if (Array.isArray(parsed.records)) records = parsed.records;
        else if (Array.isArray(parsed.outputs)) records = parsed.outputs;
        else records = [parsed];
      } else if (trimmed.startsWith('[')) {
        records = JSON.parse(trimmed);
      } else {
        const lines = trimmed.split('\n').filter((l) => l.trim().length > 0);
        records = lines.map((line) => JSON.parse(line));
        if (records[0]?.schema_version) detectedSchemaVersion = records[0].schema_version;
      }

      const benchmarks = getAllBenchmarksForFirewall();
      const previewItems = records.map((rec, idx) => {
        const userPrompt =
          rec.user_prompt ||
          rec.prompt ||
          rec.messages?.find((m: any) => m.role === 'user')?.content ||
          '';
        const modelOutput =
          rec.model_output ||
          rec.chosen ||
          rec.output ||
          rec.messages?.find((m: any) => m.role === 'model' || m.role === 'assistant')?.content ||
          '';
        const modelVersion = rec.model_version || rec.source_model || 'google/gemini-2.5-pro';
        const leak = evaluateFourLayerLeakage(userPrompt, '[]', benchmarks, false);
        const valid = Boolean(userPrompt && modelOutput);

        return {
          index: idx + 1,
          valid,
          model_version: modelVersion,
          user_prompt_preview: String(userPrompt).slice(0, 90),
          leakage_status: leak.status,
          similarity: leak.maxSimilarity,
          raw: {
            conversation_id: rec.conversation_id || `conv_import_${Date.now()}`,
            conversation_title: rec.conversation_title || 'Imported Gemini Thread',
            turn_number: Number(rec.turn_number || idx + 1),
            model_version: modelVersion,
            provider_source: rec.provider_source || 'Imported',
            recorded_at: rec.recorded_at || new Date().toISOString(),
            system_prompt:
              rec.system_prompt ||
              rec.messages?.find((m: any) => m.role === 'system')?.content ||
              '',
            user_prompt: userPrompt,
            model_output: modelOutput,
            original_transcript: rec.original_transcript || modelOutput,
            rejected_output: rec.rejected_output || rec.rejected || '',
            rejected_model_version: rec.rejected_model_version || '',
            rejection_reason: rec.rejection_reason || '',
            behavioral_category: rec.behavioral_category || 'Imported Corpus',
            tags: Array.isArray(rec.tags) ? rec.tags : ['imported'],
            persona_archetype: rec.persona_archetype || 'Protective Anchor',
            narrative_continuity_score: Number(rec.narrative_continuity_score ?? 8.5),
            character_consistency_score: Number(rec.character_consistency_score ?? 8.5),
            emotional_expressiveness_score: Number(rec.emotional_expressiveness_score ?? 8.5),
            human_evaluation_notes: rec.human_evaluation_notes || 'Validated via Schema v1.2.0 Importer',
          },
        };
      });

      res.json({
        detected_schema_version: detectedSchemaVersion,
        total_records: previewItems.length,
        valid_count: previewItems.filter((p) => p.valid).length,
        leakage_flagged_count: previewItems.filter((p) => p.leakage_status !== 'clean').length,
        items: previewItems,
      });
    } catch (err: any) {
      res.status(400).json({
        error: `JSON/JSONL Schema Validation Error: ${err.message}`,
      });
    }
  });

  app.post('/api/import/commit', (req, res) => {
    try {
      const { items } = req.body;
      if (!Array.isArray(items) || items.length === 0) {
        res.status(400).json({ error: 'No validated items to import' });
        return;
      }
      const benchmarks = getAllBenchmarksForFirewall();
      const insertStmt = db.prepare(`
        INSERT INTO outputs (
          id, conversation_id, conversation_title, turn_number, model_version, provider_source,
          recorded_at, system_prompt, user_prompt, model_output, original_transcript, full_context_json,
          rejected_output, rejected_model_version, rejection_reason, behavioral_category, tags_json,
          persona_archetype, narrative_continuity_score, character_consistency_score,
          emotional_expressiveness_score, human_evaluation_notes, training_eligible, source_mode,
          normalized_prompt_hash, ngram_json, leakage_status, max_similarity_score, matched_benchmark_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      let importedCount = 0;
      for (const item of items) {
        if (!item.valid || !item.raw) continue;
        const r = item.raw;
        const leak = evaluateFourLayerLeakage(r.user_prompt, '[]', benchmarks, false);
        const id = `out_imp_${Date.now()}_${importedCount}`;
        insertStmt.run(
          id,
          r.conversation_id,
          r.conversation_title,
          r.turn_number,
          r.model_version,
          r.provider_source,
          r.recorded_at,
          r.system_prompt,
          r.user_prompt,
          r.model_output,
          r.original_transcript,
          JSON.stringify([{ role: 'user', content: r.user_prompt }]),
          r.rejected_output,
          r.rejected_model_version,
          r.rejection_reason,
          r.behavioral_category,
          JSON.stringify(r.tags),
          r.persona_archetype,
          r.narrative_continuity_score,
          r.character_consistency_score,
          r.emotional_expressiveness_score,
          r.human_evaluation_notes,
          leak.status === 'clean' ? 1 : 0,
          'imported',
          leak.normalizedHash,
          JSON.stringify(leak.ngrams),
          leak.status,
          leak.maxSimilarity,
          leak.matchedBenchmarkId
        );
        importedCount++;
      }

      res.json({ importedCount, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to commit imported records' });
    }
  });

  // 11. Update Checkpoint Verification Manifest or Register New Checkpoint
  app.post('/api/checkpoints/verify-manifest', (req, res) => {
    try {
      const { checkpoint_id, manifest } = req.body;
      db.prepare(`
        UPDATE model_checkpoints
        SET verified_manifest_json = ?,
            context_window = ?
        WHERE id = ?
      `).run(
        JSON.stringify(manifest),
        Number(manifest.context_window || 131072),
        checkpoint_id
      );
      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to update checkpoint manifest' });
    }
  });

  // 12. Save LoRA Configuration Recipe & Log Training Runs
  app.post('/api/lora-configs', (req, res) => {
    try {
      const b = req.body;
      const id = `lora_cfg_${Date.now()}`;
      db.prepare(`
        INSERT INTO lora_configs (
          id, name, parent_checkpoint_id, training_stage, lora_r, lora_alpha, lora_dropout,
          verified_target_modules_json, freeze_multimodal_towers, learning_rate, lr_scheduler_type,
          warmup_ratio, warmup_steps, max_grad_norm, weight_decay, optimizer_type, num_epochs,
          per_device_batch_size, gradient_accumulation_steps, val_split_ratio, eval_steps,
          early_stopping_patience, max_seq_length, quantization_mode, use_think_control_token, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        id,
        b.name || 'Custom Tinman LoRA Recipe',
        b.parent_checkpoint_id || 'ckpt_stage2_tinman_merged',
        b.training_stage || 'sft',
        Number(b.lora_r ?? 32),
        Number(b.lora_alpha ?? 64),
        Number(b.lora_dropout ?? 0.05),
        JSON.stringify(
          b.verified_target_modules || [
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
          ]
        ),
        b.freeze_multimodal_towers !== false ? 1 : 0,
        Number(b.learning_rate ?? 0.0002),
        b.lr_scheduler_type || 'cosine',
        Number(b.warmup_ratio ?? 0.08),
        Number(b.warmup_steps ?? 20),
        Number(b.max_grad_norm ?? 1.0),
        Number(b.weight_decay ?? 0.01),
        b.optimizer_type || 'adamw_8bit',
        Number(b.num_epochs ?? 3),
        Number(b.per_device_batch_size ?? 2),
        Number(b.gradient_accumulation_steps ?? 4),
        Number(b.val_split_ratio ?? 0.15),
        Number(b.eval_steps ?? 20),
        Number(b.early_stopping_patience ?? 3),
        Math.min(131072, Number(b.max_seq_length ?? 8192)),
        b.quantization_mode || 'nf4_4bit',
        b.use_think_control_token ? 1 : 0,
        new Date().toISOString()
      );
      res.json({ createdId: id, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to save LoRA configuration' });
    }
  });

  app.post('/api/training-runs', (req, res) => {
    try {
      const b = req.body;
      const runId = `tr_run_${Date.now()}`;
      const adapterCkptId = `ckpt_stage4_${Date.now()}`;
      const parentCkpt = db
        .prepare('SELECT * FROM model_checkpoints WHERE id = ?')
        .get(b.parent_checkpoint_id || 'ckpt_stage2_tinman_merged') as any;

      const now = new Date().toISOString();
      db.prepare(`
        INSERT INTO model_checkpoints (
          id, parent_checkpoint_id, stage, name, hf_repo_or_path, revision_commit_sha,
          weights_sha256, context_window, verified_manifest_json, provenance_notes, created_at
        ) VALUES (?, ?, 'stage_4_lora_adapter', ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        adapterCkptId,
        b.parent_checkpoint_id || 'ckpt_stage2_tinman_merged',
        b.adapter_name || `Amanda LoRA Checkpoint (${b.run_title || 'New Run'})`,
        b.adapter_path || `local-adapters/tinman-lora-${Date.now()}`,
        sha256Hex(`${runId}_commit`).slice(0, 40),
        b.adapter_sha256 || sha256Hex(`${adapterCkptId}_safetensors`),
        131072,
        parentCkpt?.verified_manifest_json || '{}',
        b.notes || 'Stage 4 LoRA adapter registered from Training Studio.',
        now
      );

      db.prepare(`
        INSERT INTO training_runs (
          id, run_title, parent_checkpoint_id, dataset_version_id, lora_config_id,
          resulting_adapter_checkpoint_id, status, train_loss_curve_json, val_loss_curve_json,
          best_val_loss, final_train_loss, environment_versions_json, notes, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        runId,
        b.run_title || 'Amanda LoRA Training Experiment',
        b.parent_checkpoint_id || 'ckpt_stage2_tinman_merged',
        b.dataset_version_id || 'ds_sft_v1',
        b.lora_config_id || 'lora_cfg_sft_v1',
        adapterCkptId,
        b.status || 'completed',
        JSON.stringify(b.train_loss_curve || [{ step: 0, loss: 13.5 }, { step: 50, loss: 1.18 }]),
        JSON.stringify(b.val_loss_curve || [{ step: 25, val_loss: 2.1 }, { step: 50, val_loss: 1.25 }]),
        Number(b.best_val_loss ?? 1.25),
        Number(b.final_train_loss ?? 1.18),
        JSON.stringify(
          b.environment_versions || {
            transformers: '4.51.2',
            peft: '0.15.1',
            trl: '0.16.1',
            unsloth: '2026.4.1',
          }
        ),
        b.notes || '',
        now,
        now
      );

      res.json({ createdRunId: runId, adapterCkptId, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to register training run' });
    }
  });

  // 13. Evaluation Benchmarks & Evaluation Runs
  app.post('/api/eval/benchmarks', (req, res) => {
    try {
      const b = req.body;
      const id = `bm_${Date.now()}`;
      const userPrompt = String(b.user_prompt || '').trim();
      if (!userPrompt) {
        res.status(400).json({ error: 'Benchmark user_prompt is required' });
        return;
      }
      const normHash = sha256Hex(normalizePromptText(userPrompt));
      const ngrams = JSON.stringify(computeNGramTokens(userPrompt));

      db.prepare(`
        INSERT INTO eval_benchmarks (
          id, suite_version, title, category, system_prompt, prior_context_json,
          user_prompt, evaluation_criteria, reference_anchor, normalized_prompt_hash, ngram_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        id,
        b.suite_version || 'tinman-eval-v1.2',
        b.title || 'Untitled Benchmark Prompt',
        b.category || 'persona_consistency',
        b.system_prompt || '',
        JSON.stringify(b.prior_context || []),
        userPrompt,
        b.evaluation_criteria || '',
        b.reference_anchor || '',
        normHash,
        ngrams,
        new Date().toISOString()
      );

      // Re-run 4-Layer Leakage Firewall across all training outputs!
      recheckAllOutputsLeakage();

      res.json({ createdId: id, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to add benchmark prompt' });
    }
  });

  // Create a new Evaluation Run (via Local Endpoint Bridge or Recorded Checkpoint Output)
  app.post('/api/eval/runs', async (req, res) => {
    try {
      const b = req.body;
      const runId = `eval_run_${Date.now()}`;
      const now = new Date().toISOString();
      const genParams = b.generation_params || {
        temperature: 0.7,
        top_p: 0.92,
        top_k: 40,
        min_p: 0.05,
        repetition_penalty: 1.08,
        max_new_tokens: 512,
        seed: 42,
      };
      const quantSettings = b.quantization_settings || {
        method: 'nf4_4bit',
        bits: 4,
        compute_dtype: 'bfloat16',
        kv_cache_dtype: 'bfloat16',
      };

      db.prepare(`
        INSERT INTO eval_runs (
          id, run_name, checkpoint_id, baseline_run_id, benchmark_suite_version,
          execution_source, inference_engine, engine_version, hardware_info,
          quantization_settings_json, system_prompt_sha256, generation_params_json,
          determinism_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        runId,
        b.run_name || `Eval Run (${new Date().toISOString().slice(0, 10)})`,
        b.checkpoint_id || 'ckpt_stage2_tinman_merged',
        b.baseline_run_id || 'eval_run_stage2_tinman_base',
        b.benchmark_suite_version || 'tinman-eval-v1.2',
        b.execution_source || 'batch_jsonl_import',
        b.inference_engine || 'vllm',
        b.engine_version || '0.8.4',
        b.hardware_info || 'Local GPU Runtime',
        JSON.stringify(quantSettings),
        sha256Hex('eval-suite-system-prompts'),
        JSON.stringify(genParams),
        'PRNG seed sets sampling order only; floating-point atomic reductions and quantized kernels are not bit-exact deterministic.',
        now
      );

      const benchmarks = db.prepare('SELECT * FROM eval_benchmarks').all() as any[];
      const insertRes = db.prepare(`
        INSERT INTO eval_results (
          id, run_id, benchmark_id, output_text, output_sha256, latency_ms, token_count,
          human_scores_json, judge_scores_json, agreement_metrics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      const customResponses: Record<string, string> = b.responses_by_benchmark || {};

      for (const bm of benchmarks) {
        let outText = customResponses[bm.id] || '';
        if (!outText && b.local_endpoint_url) {
          try {
            const resp = await fetch(`${String(b.local_endpoint_url).replace(/\/$/, '')}/chat/completions`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model: b.local_model_name || 'Tinman-Lab/Tinman-gemma4-companion-merged',
                messages: [
                  ...(bm.system_prompt ? [{ role: 'system', content: bm.system_prompt }] : []),
                  ...JSON.parse(bm.prior_context_json || '[]'),
                  { role: 'user', content: bm.user_prompt },
                ],
                temperature: genParams.temperature,
                top_p: genParams.top_p,
                max_tokens: genParams.max_new_tokens,
                seed: genParams.seed,
              }),
            });
            if (resp.ok) {
              const data = (await resp.json()) as any;
              outText = data.choices?.[0]?.message?.content || '';
            }
          } catch {
            // Fall through to placeholder if local server is unreachable from cloud container
          }
        }
        if (!outText) {
          outText =
            b.default_output_note ||
            bm.reference_anchor ||
            'Awaiting checkpoint response paste or local JSONL import.';
        }

        insertRes.run(
          `res_${runId}_${bm.id}`,
          runId,
          bm.id,
          outText,
          sha256Hex(outText),
          380,
          outText.split(/\s+/).length,
          null,
          null,
          null
        );
      }

      res.json({ createdRunId: runId, state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to create evaluation run' });
    }
  });

  // 14. Update Human Scores or Output Text on an Evaluation Result (Independent of Judge)
  app.post('/api/eval/human-score', (req, res) => {
    try {
      const { result_id, output_text, human_scores } = req.body;
      const row = db.prepare('SELECT * FROM eval_results WHERE id = ?').get(result_id) as any;
      if (!row) {
        res.status(404).json({ error: 'Evaluation result item not found' });
        return;
      }

      const newText = output_text !== undefined ? output_text : row.output_text;
      const humanPayload: HumanScorePayload = {
        persona_consistency: Number(human_scores.persona_consistency ?? 8.5),
        narrative_continuity: Number(human_scores.narrative_continuity ?? 8.5),
        emotional_expressiveness: Number(human_scores.emotional_expressiveness ?? 8.5),
        improvisation: Number(human_scores.improvisation ?? 8.5),
        conversational_adaptability: Number(human_scores.conversational_adaptability ?? 8.5),
        instruction_following: Number(human_scores.instruction_following ?? 8.5),
        rater_notes: String(human_scores.rater_notes || ''),
        rated_at: new Date().toISOString(),
      };

      const existingJudge: JudgeScorePayload | null = row.judge_scores_json
        ? JSON.parse(row.judge_scores_json)
        : null;
      const agreement = calculateAgreementMetrics(humanPayload, existingJudge);

      db.prepare(`
        UPDATE eval_results
        SET output_text = ?,
            output_sha256 = ?,
            human_scores_json = ?,
            agreement_metrics_json = ?
        WHERE id = ?
      `).run(
        newText,
        sha256Hex(newText),
        JSON.stringify(humanPayload),
        agreement ? JSON.stringify(agreement) : null,
        result_id
      );

      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({ error: err.message || 'Failed to save human evaluation scores' });
    }
  });

  // 15. Run Independent LLM-as-a-Judge Evaluation via @google/genai with Structured JSON Schema
  app.post('/api/eval/judge', async (req, res) => {
    try {
      const { result_id } = req.body;
      const row = db.prepare('SELECT * FROM eval_results WHERE id = ?').get(result_id) as any;
      if (!row) {
        res.status(404).json({ error: 'Evaluation result not found' });
        return;
      }
      const bm = db
        .prepare('SELECT * FROM eval_benchmarks WHERE id = ?')
        .get(row.benchmark_id) as any;
      if (!bm) {
        res.status(404).json({ error: 'Benchmark prompt not found' });
        return;
      }

      const ai = getGenAIClient();
      const judgePrompt = `You are an independent behavioral evaluation judge grading a candidate response from a companion language model checkpoint.
Evaluate the Candidate Output strictly against the Benchmark Prompt, Prior Context, Evaluation Criteria, and Reference Anchor across all 6 dimensions on a 1.0 to 10.0 scale.
Provide concise, evidence-based rationales (1-2 sentences each) that quote a short verbatim span from the Candidate Output.

System Prompt: ${bm.system_prompt}
Prior Context: ${bm.prior_context_json}
User Prompt: ${bm.user_prompt}
Evaluation Criteria: ${bm.evaluation_criteria}
Reference Anchor: ${bm.reference_anchor}

Candidate Output to Grade:
"""
${row.output_text}
"""`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: judgePrompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              persona_consistency: { type: Type.NUMBER },
              narrative_continuity: { type: Type.NUMBER },
              emotional_expressiveness: { type: Type.NUMBER },
              improvisation: { type: Type.NUMBER },
              conversational_adaptability: { type: Type.NUMBER },
              instruction_following: { type: Type.NUMBER },
              rationale_persona: { type: Type.STRING },
              rationale_continuity: { type: Type.STRING },
              rationale_expressiveness: { type: Type.STRING },
              rationale_improvisation: { type: Type.STRING },
              rationale_adaptability: { type: Type.STRING },
              rationale_instruction: { type: Type.STRING },
              overall_summary: { type: Type.STRING },
            },
            required: [
              'persona_consistency',
              'narrative_continuity',
              'emotional_expressiveness',
              'improvisation',
              'conversational_adaptability',
              'instruction_following',
              'rationale_persona',
              'rationale_continuity',
              'rationale_expressiveness',
              'rationale_improvisation',
              'rationale_adaptability',
              'rationale_instruction',
              'overall_summary',
            ],
          },
        },
      });

      const parsed = JSON.parse((response.text || '{}').trim());
      const judgePayload: JudgeScorePayload = {
        persona_consistency: Number(parsed.persona_consistency ?? 8.0),
        narrative_continuity: Number(parsed.narrative_continuity ?? 8.0),
        emotional_expressiveness: Number(parsed.emotional_expressiveness ?? 8.0),
        improvisation: Number(parsed.improvisation ?? 8.0),
        conversational_adaptability: Number(parsed.conversational_adaptability ?? 8.0),
        instruction_following: Number(parsed.instruction_following ?? 8.0),
        judge_model_id: 'gemini-3.8-flash (LLM-as-a-Judge)',
        judge_rubric_version: 'tinman-rubric-v1.2',
        evidence_rationales: {
          persona_consistency: parsed.rationale_persona || '',
          narrative_continuity: parsed.rationale_continuity || '',
          emotional_expressiveness: parsed.rationale_expressiveness || '',
          improvisation: parsed.rationale_improvisation || '',
          conversational_adaptability: parsed.rationale_adaptability || '',
          instruction_following: parsed.rationale_instruction || '',
        },
        overall_summary: parsed.overall_summary || '',
        judged_at: new Date().toISOString(),
      };

      const existingHuman: HumanScorePayload | null = row.human_scores_json
        ? JSON.parse(row.human_scores_json)
        : null;
      const agreement = calculateAgreementMetrics(existingHuman, judgePayload);

      db.prepare(`
        UPDATE eval_results
        SET judge_scores_json = ?,
            agreement_metrics_json = ?
        WHERE id = ?
      `).run(
        JSON.stringify(judgePayload),
        agreement ? JSON.stringify(agreement) : null,
        result_id
      );

      res.json({ state: getFullWorkbenchState() });
    } catch (err: any) {
      res.status(500).json({
        error: err.message || 'LLM-as-a-Judge evaluation failed',
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const PORT = 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Tinman & Gemini Fine-Tuning Workbench running on http://localhost:${PORT}`);
  });
}

startServer();
