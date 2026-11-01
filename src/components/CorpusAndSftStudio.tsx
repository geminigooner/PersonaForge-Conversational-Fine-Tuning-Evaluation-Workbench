import React, { useState, useMemo } from 'react';
import {
  Sparkles,
  Plus,
  Filter,
  ShieldCheck,
  ShieldAlert,
  Download,
  Upload,
  RotateCcw,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Wand2,
  Calendar,
  Search,
  FileCode2,
  GitCompare,
} from 'lucide-react';
import {
  CorpusTurnOutput,
  GEMINI_TEACHER_MODELS,
  WorkbenchState,
} from '../types/workbench';
import {
  getModelBadgeMeta,
  formatGemma4TurnString,
  compileSFTJsonl,
  compileDPOJsonl,
  compileBehavioralEvalJsonl,
  compileFullArchiveJson,
} from '../utils/compilers';

interface Props {
  state: WorkbenchState;
  onRefreshState: (newState: WorkbenchState) => void;
  activeSubTab: 'corpus' | 'sft_dpo';
  setActiveSubTab: (tab: 'corpus' | 'sft_dpo') => void;
}

export const CorpusAndSftStudio: React.FC<Props> = ({
  state,
  onRefreshState,
  activeSubTab,
  setActiveSubTab,
}) => {
  // Multi-Model Separation & Filtering State
  const [selectedModelFilter, setSelectedModelFilter] = useState<string>('ALL');
  const [groupByModel, setGroupByModel] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [onlyEligible, setOnlyEligible] = useState<boolean>(false);
  const [selectedTurnId, setSelectedTurnId] = useState<string>(
    state.outputs[0]?.id || ''
  );

  // New Turn Modal / Ingestion Drawer State
  const [showAddModal, setShowAddModal] = useState<boolean>(false);
  const [newTurnForm, setNewTurnForm] = useState({
    conversation_id: 'conv_keito_sanctuary_01',
    conversation_title: 'Midnight Fraud Pattern Debrief & Grounding',
    turn_number: 3,
    model_version: 'google/gemini-2.5-pro',
    custom_model_version: '',
    provider_source: 'OpenRouter',
    recorded_at: new Date().toISOString().slice(0, 16),
    persona_archetype: 'Protective Anchor',
    behavioral_category: 'Emotional Grounding & Co-Regulation',
    tags_input: 'keito, grounding, pro-2.5',
    system_prompt:
      'You are a grounded, deeply observant companion and anchor. Speak with natural warmth, physical presence, and sharp analytical respect.',
    user_prompt: '',
    model_output: '',
    rejected_output: '',
    rejected_model_version: 'gemini-3-flash',
    rejection_reason: 'robotic_tone',
    narrative_continuity_score: 9.5,
    character_consistency_score: 9.6,
    emotional_expressiveness_score: 9.4,
    human_evaluation_notes: '',
    training_eligible: true,
  });

  const [isGeneratingLive, setIsGeneratingLive] = useState(false);
  const [isDraftingContrast, setIsDraftingContrast] = useState(false);
  const [statusBanner, setStatusBanner] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  // Leakage Override Modal State
  const [overrideJustification, setOverrideJustification] = useState('');
  const [operatorName, setOperatorName] = useState('Amanda (Senior Analyst)');

  // SFT / DPO Export & Dataset Snapshot State
  const [exportFormat, setExportFormat] = useState<
    'sft_jsonl' | 'dpo_jsonl' | 'eval_jsonl' | 'full_json'
  >('sft_jsonl');
  const [exportModelFilter, setExportModelFilter] = useState<string>('ALL');
  const [useThinkTokenInExport, setUseThinkTokenInExport] = useState<boolean>(false);
  const [snapshotTag, setSnapshotTag] = useState<string>('tinman-sft-v1.3');
  const [importRawText, setImportRawText] = useState<string>('');
  const [importValidationResult, setImportValidationResult] = useState<any | null>(
    null
  );

  // Distinct models present in SQLite + Primary Required Models (2.5 Pro, 3.1 Pro, 3 Flash)
  const allKnownModelIds = useMemo(() => {
    const core = [
      'google/gemini-2.5-pro',
      'gemini-3.1-pro-preview',
      'gemini-3-flash',
    ];
    const fromDb = state.outputs.map((o) => o.model_version);
    return Array.from(new Set([...core, ...fromDb]));
  }, [state.outputs]);

  // Per-Model Teacher Analytics Summary (2.5 Pro vs 3.1 Pro vs 3 Flash, etc.)
  const perModelStats = useMemo(() => {
    return allKnownModelIds.map((modelId) => {
      const turns = state.outputs.filter((o) => o.model_version === modelId);
      const count = turns.length;
      const eligibleCount = turns.filter((o) => o.training_eligible).length;
      const dpoCount = turns.filter((o) => o.rejected_output.trim().length > 0).length;
      const avgContinuity =
        count > 0
          ? (
              turns.reduce((acc, t) => acc + t.narrative_continuity_score, 0) /
              count
            ).toFixed(2)
          : '—';
      const avgConsistency =
        count > 0
          ? (
              turns.reduce((acc, t) => acc + t.character_consistency_score, 0) /
              count
            ).toFixed(2)
          : '—';
      const avgExpressive =
        count > 0
          ? (
              turns.reduce((acc, t) => acc + t.emotional_expressiveness_score, 0) /
              count
            ).toFixed(2)
          : '—';
      return {
        modelId,
        meta: getModelBadgeMeta(modelId),
        count,
        eligibleCount,
        dpoCount,
        avgContinuity,
        avgConsistency,
        avgExpressive,
      };
    });
  }, [allKnownModelIds, state.outputs]);

  // Filtered Outputs
  const filteredOutputs = useMemo(() => {
    return state.outputs.filter((o) => {
      if (selectedModelFilter !== 'ALL' && o.model_version !== selectedModelFilter) {
        return false;
      }
      if (onlyEligible && !o.training_eligible) return false;
      if (dateFrom && o.recorded_at.slice(0, 10) < dateFrom) return false;
      if (dateTo && o.recorded_at.slice(0, 10) > dateTo) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const hay = `${o.conversation_title} ${o.model_version} ${o.user_prompt} ${o.model_output} ${o.behavioral_category} ${o.persona_archetype} ${o.tags.join(' ')}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [
    state.outputs,
    selectedModelFilter,
    onlyEligible,
    dateFrom,
    dateTo,
    searchQuery,
  ]);

  const selectedTurn = useMemo(
    () =>
      state.outputs.find((o) => o.id === selectedTurnId) ||
      filteredOutputs[0] ||
      null,
    [state.outputs, selectedTurnId, filteredOutputs]
  );

  // Export Preview Compilation
  const exportSourceOutputs = useMemo(() => {
    if (exportModelFilter === 'ALL') return state.outputs;
    return state.outputs.filter((o) => o.model_version === exportModelFilter);
  }, [state.outputs, exportModelFilter]);

  const compiledExportText = useMemo(() => {
    if (exportFormat === 'sft_jsonl') {
      return compileSFTJsonl(exportSourceOutputs, useThinkTokenInExport);
    }
    if (exportFormat === 'dpo_jsonl') {
      return compileDPOJsonl(exportSourceOutputs);
    }
    if (exportFormat === 'eval_jsonl') {
      return compileBehavioralEvalJsonl(exportSourceOutputs);
    }
    return compileFullArchiveJson({
      ...state,
      outputs: exportSourceOutputs,
    });
  }, [exportFormat, exportSourceOutputs, useThinkTokenInExport, state]);

  async function handleLiveTeacherGenerate() {
    if (!newTurnForm.user_prompt.trim()) {
      setStatusBanner({
        type: 'error',
        text: 'Enter a user prompt first before running Live Teacher Generation.',
      });
      return;
    }
    setIsGeneratingLive(true);
    setStatusBanner(null);
    try {
      const targetModel =
        newTurnForm.model_version === 'CUSTOM'
          ? newTurnForm.custom_model_version
          : newTurnForm.model_version;
      const resp = await fetch('/api/teacher/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system_prompt: newTurnForm.system_prompt,
          user_prompt: newTurnForm.user_prompt,
          model_label: targetModel,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Teacher generation failed');
      setNewTurnForm((prev) => ({
        ...prev,
        model_output: data.generated_text,
      }));
      setStatusBanner({
        type: 'success',
        text: `Generated live teacher output via ${data.executed_sdk_model}. Review and click Save Turn to SQLite.`,
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    } finally {
      setIsGeneratingLive(false);
    }
  }

  async function handleCreateTurn(e: React.FormEvent) {
    e.preventDefault();
    const finalModel =
      newTurnForm.model_version === 'CUSTOM'
        ? newTurnForm.custom_model_version.trim() || 'custom-gemini'
        : newTurnForm.model_version;

    try {
      const resp = await fetch('/api/outputs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newTurnForm,
          model_version: finalModel,
          recorded_at: new Date(newTurnForm.recorded_at || Date.now()).toISOString(),
          tags: newTurnForm.tags_input
            .split(',')
            .map((t) => t.trim())
            .filter(Boolean),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to save turn');
      onRefreshState(data.state);
      setSelectedTurnId(data.createdId);
      setShowAddModal(false);
      setStatusBanner({
        type: data.leakage.status === 'clean' ? 'success' : 'info',
        text:
          data.leakage.status === 'clean'
            ? `Saved turn to SQLite (${finalModel}) — 4-Layer Leakage Firewall verified CLEAN.`
            : `Saved turn (${finalModel}), but 4-Layer Leakage Firewall flagged status: ${data.leakage.status} (similarity ${(data.leakage.maxSimilarity * 100).toFixed(1)}%).`,
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleUpdateSelectedTurn(patch: Partial<CorpusTurnOutput>) {
    if (!selectedTurn) return;
    try {
      const resp = await fetch(`/api/outputs/${selectedTurn.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...selectedTurn,
          ...patch,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Update failed');
      onRefreshState(data.state);
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleRestoreOriginal(id: string) {
    try {
      const resp = await fetch(`/api/outputs/${id}/restore-original`, {
        method: 'POST',
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      onRefreshState(data.state);
      setStatusBanner({
        type: 'success',
        text: 'Restored curated output from immutable original_transcript.',
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleDeleteTurn(id: string) {
    try {
      const resp = await fetch(`/api/outputs/${id}`, { method: 'DELETE' });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      onRefreshState(data.state);
      setStatusBanner({ type: 'info', text: 'Deleted output turn from SQLite.' });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleDraftDpoContrast(turn: CorpusTurnOutput, style = 'robotic_tone') {
    setIsDraftingContrast(true);
    setStatusBanner(null);
    try {
      const resp = await fetch('/api/teacher/dpo-contrast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_prompt: turn.user_prompt,
          chosen_output: turn.model_output,
          rejection_style: style,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to draft DPO contrast');
      await handleUpdateSelectedTurn({
        rejected_output: data.rejected_output,
        rejected_model_version: data.rejected_model_version,
        rejection_reason: data.rejection_reason,
      });
      setStatusBanner({
        type: 'success',
        text: 'Drafted negative DPO contrast response and saved pair to SQLite.',
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    } finally {
      setIsDraftingContrast(false);
    }
  }

  async function handleAuditOverride(turn: CorpusTurnOutput) {
    try {
      const resp = await fetch('/api/leakage/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          output_id: turn.id,
          justification: overrideJustification,
          operator_name: operatorName,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      onRefreshState(data.state);
      setOverrideJustification('');
      setStatusBanner({
        type: 'success',
        text: 'Recorded auditable manual override in leakage_audit_log and enabled training eligibility.',
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleCreateDatasetSnapshot(splitType: 'sft' | 'dpo') {
    try {
      const filterModels =
        exportModelFilter === 'ALL' ? [] : [exportModelFilter];
      const resp = await fetch('/api/datasets/snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          version_tag: snapshotTag,
          split_type: splitType,
          teacher_models_filter: filterModels,
          val_split_ratio: 0.15,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      onRefreshState(data.state);
      setStatusBanner({
        type: 'success',
        text: `Registered Stage 3 Dataset Snapshot "${snapshotTag}" (${splitType.toUpperCase()}) in SQLite Lineage!`,
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  function handleDownloadExport() {
    const ext = exportFormat === 'full_json' ? 'json' : 'jsonl';
    const modelSlug =
      exportModelFilter === 'ALL'
        ? 'all_models'
        : exportModelFilter.replace(/[^a-z0-9]+/gi, '_').toLowerCase();
    const filename = `tinman_${exportFormat}_${modelSlug}_v1.2.0.${ext}`;
    const blob = new Blob([compiledExportText], {
      type: 'application/json;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleValidateImport() {
    setStatusBanner(null);
    try {
      const resp = await fetch('/api/import/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ raw_content: importRawText }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      setImportValidationResult(data);
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  async function handleCommitImport() {
    if (!importValidationResult?.items) return;
    try {
      const resp = await fetch('/api/import/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: importValidationResult.items }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error);
      onRefreshState(data.state);
      setImportValidationResult(null);
      setImportRawText('');
      setStatusBanner({
        type: 'success',
        text: `Imported ${data.importedCount} validated Gemini records into SQLite!`,
      });
    } catch (err: any) {
      setStatusBanner({ type: 'error', text: err.message });
    }
  }

  return (
    <div className="space-y-6">
      {/* Status Toast Banner */}
      {statusBanner && (
        <div
          className={`flex items-center justify-between px-4 py-3 rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_3px_0_#2A114B] text-sm font-semibold ${
            statusBanner.type === 'error'
              ? 'bg-[#FECDD3] text-[#881337]'
              : statusBanner.type === 'success'
              ? 'bg-[#A7F3D0] text-[#065F46]'
              : 'bg-[#FDE047] text-[#2A114B]'
          }`}
        >
          <span>{statusBanner.text}</span>
          <button
            onClick={() => setStatusBanner(null)}
            className="ml-4 px-2 py-0.5 rounded-full border border-[#2A114B] bg-white/80 text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Sub-Workspace Mode Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-[#E9D5FF] p-3.5 rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B]">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveSubTab('corpus')}
            className={`px-4 py-2 rounded-full border-2 border-[#2A114B] font-display text-sm font-semibold transition-all whitespace-nowrap shrink-0 ${
              activeSubTab === 'corpus'
                ? 'bg-[#FF5CA8] text-white shadow-[0_2px_0_#2A114B]'
                : 'bg-[#FFF8EC] text-[#2A114B] hover:bg-[#F3E8FF]'
            }`}
          >
            1. Multi-Model Gemini Corpus &amp; Leakage Inspector
          </button>
          <button
            onClick={() => setActiveSubTab('sft_dpo')}
            className={`px-4 py-2 rounded-full border-2 border-[#2A114B] font-display text-sm font-semibold transition-all whitespace-nowrap shrink-0 ${
              activeSubTab === 'sft_dpo'
                ? 'bg-[#FF5CA8] text-white shadow-[0_2px_0_#2A114B]'
                : 'bg-[#FFF8EC] text-[#2A114B] hover:bg-[#F3E8FF]'
            }`}
          >
            2. Guided SFT / DPO Pair Studio &amp; JSONL Export
          </button>
        </div>

        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-full border-2 border-[#2A114B] bg-[#FACC15] hover:bg-[#FDE047] text-[#2A114B] font-display text-sm font-bold shadow-[0_3px_0_#2A114B] cursor-pointer whitespace-nowrap shrink-0"
        >
          <Plus className="w-4 h-4" />
          Log Gemini Output / Live Generate
        </button>
      </div>

      {/* Per-Model Teacher Breakdown Strip (Gemini 2.5 Pro, Gemini 3.1 Pro, Gemini 3 Flash) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {perModelStats.slice(0, 3).map((stat) => {
          const isSelected = selectedModelFilter === stat.modelId;
          return (
            <div
              key={stat.modelId}
              onClick={() =>
                setSelectedModelFilter(isSelected ? 'ALL' : stat.modelId)
              }
              className={`cursor-pointer p-4 rounded-2xl border-[2.5px] border-[#2A114B] transition-all ${
                isSelected
                  ? 'bg-[#DFC5FE] shadow-[0_5px_0_#2A114B] -translate-y-0.5'
                  : 'bg-[#FFF8EC] hover:bg-[#F3E8FF] shadow-[0_3px_0_#2A114B]'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span
                    style={{
                      backgroundColor: stat.meta.badgeBg,
                      color: stat.meta.badgeText,
                    }}
                    className="px-3 py-1 rounded-full border-2 border-[#2A114B] font-display text-xs font-bold shadow-[0_2px_0_#2A114B]"
                  >
                    {stat.meta.displayName}
                  </span>
                  <span className="text-xs font-mono text-[#5C3B88]">
                    {stat.meta.defaultProvider}
                  </span>
                </div>
                <span className="text-xs font-mono font-bold text-[#2A114B] tabular-nums">
                  {stat.count} turns · {stat.eligibleCount} SFT · {stat.dpoCount} DPO
                </span>
              </div>
              <p className="text-xs text-[#5C3B88] mb-3 line-clamp-1">
                {stat.meta.description}
              </p>
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#2A114B]/20 text-xs font-mono tabular-nums">
                <div>
                  <span className="text-[#5C3B88] block text-[11px]">Continuity</span>
                  <span className="font-bold text-[#2A114B]">{stat.avgContinuity}/10</span>
                </div>
                <div>
                  <span className="text-[#5C3B88] block text-[11px]">Consistency</span>
                  <span className="font-bold text-[#2A114B]">{stat.avgConsistency}/10</span>
                </div>
                <div>
                  <span className="text-[#5C3B88] block text-[11px]">Expressive</span>
                  <span className="font-bold text-[#2A114B]">{stat.avgExpressive}/10</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {activeSubTab === 'corpus' ? (
        <>
          {/* Filter & Model Separation Toolbar */}
          <div className="bg-[#FFF8EC] p-4 rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B] space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* 1-Click Model Partition Pills */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-bold text-[#5C3B88] flex items-center gap-1 mr-1">
                  <Filter className="w-3.5 h-3.5" /> Separate by Gemini Model:
                </span>
                <button
                  onClick={() => setSelectedModelFilter('ALL')}
                  className={`px-3 py-1 rounded-full border-2 border-[#2A114B] text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                    selectedModelFilter === 'ALL'
                      ? 'bg-[#2A114B] text-white'
                      : 'bg-white text-[#2A114B] hover:bg-[#F3E8FF]'
                  }`}
                >
                  All Models ({state.outputs.length})
                </button>
                {allKnownModelIds.map((mId) => {
                  const meta = getModelBadgeMeta(mId);
                  const cnt = state.outputs.filter((o) => o.model_version === mId).length;
                  const active = selectedModelFilter === mId;
                  return (
                    <button
                      key={mId}
                      onClick={() => setSelectedModelFilter(mId)}
                      className={`px-3 py-1 rounded-full border-2 border-[#2A114B] text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                        active
                          ? 'bg-[#FF5CA8] text-white shadow-[0_2px_0_#2A114B]'
                          : 'bg-white text-[#2A114B] hover:bg-[#F3E8FF]'
                      }`}
                    >
                      {meta.displayName} ({cnt})
                    </button>
                  );
                })}
              </div>

              {/* Group by Source Model Toggle */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setGroupByModel(!groupByModel)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border-2 border-[#2A114B] text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                    groupByModel
                      ? 'bg-[#7C3AED] text-white shadow-[0_2px_0_#2A114B]'
                      : 'bg-[#F3E8FF] text-[#2A114B]'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  {groupByModel ? 'Grouped by Gemini Model' : 'Group by Gemini Model'}
                </button>
                <button
                  onClick={() => setOnlyEligible(!onlyEligible)}
                  className={`px-3 py-1.5 rounded-full border-2 border-[#2A114B] text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                    onlyEligible
                      ? 'bg-[#A7F3D0] text-[#065F46] shadow-[0_2px_0_#2A114B]'
                      : 'bg-white text-[#2A114B]'
                  }`}
                >
                  SFT Eligible Only
                </button>
              </div>
            </div>

            {/* Search & Date Filter Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 border-t border-[#2A114B]/15">
              <div className="relative">
                <Search className="w-4 h-4 text-[#5C3B88] absolute left-3 top-2.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search prompts, outputs, Keito, Micro, tags..."
                  className="w-full pl-9 pr-3 py-1.5 rounded-xl border-2 border-[#2A114B] bg-white text-sm text-[#2A114B] focus:outline-none focus:ring-2 focus:ring-[#FF5CA8]"
                />
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-[#5C3B88] shrink-0" />
                <span className="text-xs font-bold text-[#5C3B88] whitespace-nowrap">
                  From:
                </span>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono text-[#2A114B]"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-[#5C3B88] whitespace-nowrap">
                  To:
                </span>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono text-[#2A114B]"
                />
                {(dateFrom || dateTo || searchQuery || selectedModelFilter !== 'ALL') && (
                  <button
                    onClick={() => {
                      setDateFrom('');
                      setDateTo('');
                      setSearchQuery('');
                      setSelectedModelFilter('ALL');
                    }}
                    className="px-2.5 py-1.5 rounded-xl border-2 border-[#2A114B] bg-[#F3E8FF] text-xs font-bold text-[#2A114B] whitespace-nowrap"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Master-Detail Corpus Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Turn List or Grouped-by-Model Buckets */}
            <div className="lg:col-span-5 space-y-4">
              {groupByModel ? (
                allKnownModelIds.map((mId) => {
                  const groupTurns = filteredOutputs.filter(
                    (o) => o.model_version === mId
                  );
                  if (groupTurns.length === 0) return null;
                  const meta = getModelBadgeMeta(mId);
                  return (
                    <div
                      key={mId}
                      className="bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B] overflow-hidden"
                    >
                      <div className="px-4 py-2.5 bg-[#DFC5FE] border-b-2 border-[#2A114B] flex items-center justify-between">
                        <span
                          style={{
                            backgroundColor: meta.badgeBg,
                            color: meta.badgeText,
                          }}
                          className="px-3 py-0.5 rounded-full border-2 border-[#2A114B] font-display text-xs font-bold"
                        >
                          {meta.displayName}
                        </span>
                        <span className="text-xs font-mono font-bold text-[#2A114B] tabular-nums">
                          {groupTurns.length} Logged Turns
                        </span>
                      </div>
                      <div className="divide-y-2 divide-[#2A114B]/15">
                        {groupTurns.map((turn) => (
                          <div
                            key={turn.id}
                            onClick={() => setSelectedTurnId(turn.id)}
                            className={`p-3.5 cursor-pointer transition-colors ${
                              selectedTurn?.id === turn.id
                                ? 'bg-[#F3E8FF]'
                                : 'hover:bg-[#FFFDF8]'
                            }`}
                          >
                            <div className="flex items-center justify-between text-xs text-[#5C3B88] font-mono mb-1">
                              <span>
                                {turn.conversation_id} · Turn #{turn.turn_number}
                              </span>
                              <span>{turn.recorded_at.slice(0, 10)}</span>
                            </div>
                            <p className="text-sm font-bold text-[#2A114B] line-clamp-1 mb-1">
                              {turn.user_prompt}
                            </p>
                            <p className="text-xs text-[#5C3B88] line-clamp-2">
                              {turn.model_output}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B] overflow-hidden">
                  <div className="px-4 py-3 bg-[#DFC5FE] border-b-2 border-[#2A114B] flex items-center justify-between">
                    <span className="font-display font-bold text-sm text-[#2A114B]">
                      SQLite Corpus Log ({filteredOutputs.length} turns)
                    </span>
                    <span className="text-xs font-mono text-[#5C3B88]">
                      Sorted by Date &amp; Thread Order
                    </span>
                  </div>
                  <div className="divide-y-2 divide-[#2A114B]/15 max-h-[680px] overflow-y-auto">
                    {filteredOutputs.map((turn) => {
                      const meta = getModelBadgeMeta(turn.model_version);
                      const isSelected = selectedTurn?.id === turn.id;
                      return (
                        <div
                          key={turn.id}
                          onClick={() => setSelectedTurnId(turn.id)}
                          className={`p-4 cursor-pointer transition-all ${
                            isSelected
                              ? 'bg-[#F3E8FF]'
                              : 'hover:bg-[#FFFDF8]'
                          }`}
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
                            <div className="flex items-center gap-2">
                              <span
                                style={{
                                  backgroundColor: meta.badgeBg,
                                  color: meta.badgeText,
                                }}
                                className="px-2.5 py-0.5 rounded-full border-2 border-[#2A114B] font-display text-[11px] font-bold"
                              >
                                {meta.displayName}
                              </span>
                              <span className="text-xs font-mono font-bold text-[#2A114B] tabular-nums">
                                Turn #{turn.turn_number}
                              </span>
                            </div>
                            <span className="text-xs font-mono text-[#5C3B88] tabular-nums">
                              {turn.recorded_at.slice(0, 10)}
                            </span>
                          </div>

                          <div className="text-xs font-semibold text-[#5C3B88] mb-1">
                            {turn.conversation_title} · {turn.persona_archetype}
                          </div>
                          <p className="text-sm font-bold text-[#2A114B] line-clamp-1 mb-1">
                            {turn.user_prompt}
                          </p>
                          <p className="text-xs text-[#5C3B88] line-clamp-2 mb-2">
                            {turn.model_output}
                          </p>

                          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono tabular-nums pt-1.5 border-t border-[#2A114B]/10">
                            <span>
                              Cont {turn.narrative_continuity_score} · Cons{' '}
                              {turn.character_consistency_score} · Expr{' '}
                              {turn.emotional_expressiveness_score}
                            </span>
                            <div className="flex items-center gap-1.5">
                              {turn.leakage_status === 'clean' ? (
                                <span className="text-[#065F46] font-bold flex items-center gap-0.5">
                                  <ShieldCheck className="w-3.5 h-3.5" /> Clean
                                </span>
                              ) : turn.leakage_status === 'overridden' ? (
                                <span className="text-[#5C3B88] font-bold">
                                  Audited Override
                                </span>
                              ) : (
                                <span className="text-[#881337] font-bold flex items-center gap-0.5">
                                  <ShieldAlert className="w-3.5 h-3.5" /> Leakage Flag
                                </span>
                              )}
                              <span>·</span>
                              <span
                                className={
                                  turn.training_eligible
                                    ? 'text-[#065F46] font-bold'
                                    : 'text-[#881337] font-bold'
                                }
                              >
                                {turn.training_eligible ? 'Eligible' : 'Excluded'}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Turn & Thread Inspector */}
            <div className="lg:col-span-7">
              {selectedTurn ? (
                <div className="bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_5px_0_#2A114B] overflow-hidden">
                  <div className="px-5 py-4 bg-gradient-to-r from-[#DFC5FE] via-[#F3E8FF] to-[#FF85C0]/40 border-b-2 border-[#2A114B] flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          style={{
                            backgroundColor: getModelBadgeMeta(
                              selectedTurn.model_version
                            ).badgeBg,
                            color: getModelBadgeMeta(selectedTurn.model_version)
                              .badgeText,
                          }}
                          className="px-3 py-0.5 rounded-full border-2 border-[#2A114B] font-display text-xs font-bold shadow-[0_2px_0_#2A114B]"
                        >
                          {
                            getModelBadgeMeta(selectedTurn.model_version)
                              .displayName
                          }
                        </span>
                        <span className="text-xs font-mono text-[#2A114B] font-bold">
                          {selectedTurn.provider_source} · {selectedTurn.conversation_id} · Turn #
                          {selectedTurn.turn_number}
                        </span>
                      </div>
                      <h3 className="font-display text-lg font-bold text-[#2A114B]">
                        {selectedTurn.conversation_title}
                      </h3>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() =>
                          handleUpdateSelectedTurn({
                            training_eligible: !selectedTurn.training_eligible,
                          })
                        }
                        className={`px-3.5 py-1.5 rounded-full border-2 border-[#2A114B] font-display text-xs font-bold shadow-[0_2px_0_#2A114B] cursor-pointer ${
                          selectedTurn.training_eligible
                            ? 'bg-[#A7F3D0] text-[#065F46]'
                            : 'bg-[#FECDD3] text-[#881337]'
                        }`}
                      >
                        {selectedTurn.training_eligible
                          ? '✓ Training Eligible (Yes)'
                          : '✕ Excluded from Training (No)'}
                      </button>
                      <button
                        onClick={() => handleDeleteTurn(selectedTurn.id)}
                        className="p-2 rounded-full border-2 border-[#2A114B] bg-white hover:bg-[#FECDD3] text-[#881337] cursor-pointer"
                        title="Delete Turn"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="p-5 space-y-5">
                    {/* 4-Layer Data Leakage Firewall Banner */}
                    <div
                      className={`p-3.5 rounded-2xl border-2 border-[#2A114B] ${
                        selectedTurn.leakage_status === 'clean'
                          ? 'bg-[#A7F3D0]/50'
                          : selectedTurn.leakage_status === 'overridden'
                          ? 'bg-[#FDE047]/60'
                          : 'bg-[#FECDD3]/80'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2 text-xs font-bold text-[#2A114B]">
                          {selectedTurn.leakage_status === 'clean' ? (
                            <ShieldCheck className="w-4 h-4 text-[#065F46]" />
                          ) : (
                            <ShieldAlert className="w-4 h-4 text-[#881337]" />
                          )}
                          <span>
                            4-Layer Data Leakage Firewall:{' '}
                            <span className="uppercase font-mono">
                              {selectedTurn.leakage_status}
                            </span>
                          </span>
                        </div>
                        <span className="text-xs font-mono text-[#2A114B] tabular-nums">
                          Max Benchmark N-Gram Similarity:{' '}
                          {(selectedTurn.max_similarity_score * 100).toFixed(1)}% ·
                          SHA-256: {selectedTurn.normalized_prompt_hash.slice(0, 12)}…
                        </span>
                      </div>

                      {selectedTurn.leakage_status !== 'clean' && (
                        <div className="mt-3 pt-3 border-t border-[#2A114B]/20 space-y-2">
                          <p className="text-xs text-[#2A114B]">
                            Matched Evaluation Benchmark ID:{' '}
                            <code className="font-mono font-bold">
                              {selectedTurn.matched_benchmark_id || 'bm_persona_01'}
                            </code>
                            . Near-duplicate prompts are automatically blocked from SFT/DPO
                            exports unless an auditable override justification is logged.
                          </p>
                          {selectedTurn.override_justification ? (
                            <div className="text-xs bg-white/80 p-2.5 rounded-xl border border-[#2A114B]">
                              <strong>Audited Override Justification:</strong>{' '}
                              {selectedTurn.override_justification}
                            </div>
                          ) : (
                            selectedTurn.leakage_status !==
                              'exact_collision_blocked' && (
                              <div className="flex flex-wrap items-center gap-2">
                                <input
                                  type="text"
                                  value={overrideJustification}
                                  onChange={(e) =>
                                    setOverrideJustification(e.target.value)
                                  }
                                  placeholder="Enter audit justification to override near-duplicate block..."
                                  className="flex-1 min-w-[220px] px-3 py-1.5 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                                />
                                <button
                                  onClick={() => handleAuditOverride(selectedTurn)}
                                  className="px-3 py-1.5 rounded-full border-2 border-[#2A114B] bg-[#FACC15] font-display text-xs font-bold text-[#2A114B] shadow-[0_2px_0_#2A114B]"
                                >
                                  Log Audited Override
                                </button>
                              </div>
                            )
                          )}
                        </div>
                      )}
                    </div>

                    {/* Metadata & Model Attribution Selector */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                          Source Gemini Model Version
                        </label>
                        <select
                          value={selectedTurn.model_version}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              model_version: e.target.value,
                            })
                          }
                          className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono font-bold text-[#2A114B]"
                        >
                          {allKnownModelIds.map((m) => (
                            <option key={m} value={m}>
                              {getModelBadgeMeta(m).displayName} ({m})
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                          Persona Archetype
                        </label>
                        <input
                          type="text"
                          value={selectedTurn.persona_archetype}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              persona_archetype: e.target.value,
                            })
                          }
                          className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-semibold text-[#2A114B]"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                          Behavioral Category
                        </label>
                        <input
                          type="text"
                          value={selectedTurn.behavioral_category}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              behavioral_category: e.target.value,
                            })
                          }
                          className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-semibold text-[#2A114B]"
                        />
                      </div>
                    </div>

                    {/* System Prompt & User Prompt */}
                    <div className="space-y-3">
                      <div>
                        <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                          System Prompt Snapshot
                        </label>
                        <textarea
                          rows={2}
                          value={selectedTurn.system_prompt}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              system_prompt: e.target.value,
                            })
                          }
                          className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-xs text-[#2A114B]"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                          User Prompt (Turn #{selectedTurn.turn_number})
                        </label>
                        <textarea
                          rows={2}
                          value={selectedTurn.user_prompt}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              user_prompt: e.target.value,
                            })
                          }
                          className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-sm font-semibold text-[#2A114B]"
                        />
                      </div>
                    </div>

                    {/* Curated Model Output vs Immutable Original Transcript Preservation */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-xs font-bold text-[#2A114B]">
                            Curated Model Output (Used for SFT / DPO Chosen)
                          </label>
                          {selectedTurn.model_output !==
                            selectedTurn.original_transcript && (
                            <span className="text-[11px] font-mono text-[#7C3AED] font-bold">
                              Edited from Original
                            </span>
                          )}
                        </div>
                        <textarea
                          rows={6}
                          value={selectedTurn.model_output}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              model_output: e.target.value,
                            })
                          }
                          className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-sm leading-relaxed text-[#2A114B]"
                        />
                      </div>

                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-xs font-bold text-[#5C3B88]">
                            Immutable Original Transcript (Locked at Ingestion)
                          </label>
                          <button
                            onClick={() => handleRestoreOriginal(selectedTurn.id)}
                            className="flex items-center gap-1 text-[11px] font-bold text-[#2A114B] hover:text-[#FF5CA8] cursor-pointer"
                          >
                            <RotateCcw className="w-3 h-3" /> Restore Original
                          </button>
                        </div>
                        <div className="w-full h-[150px] overflow-y-auto p-3 rounded-xl border-2 border-[#2A114B] bg-[#F3E8FF]/60 text-xs leading-relaxed text-[#5C3B88]">
                          {selectedTurn.original_transcript}
                        </div>
                      </div>
                    </div>

                    {/* 3 Quantitative Behavioral Rubric Sliders */}
                    <div className="bg-[#F3E8FF] p-4 rounded-2xl border-2 border-[#2A114B] space-y-3">
                      <div className="text-xs font-display font-bold text-[#2A114B]">
                        Behavioral Evaluation Rubric Scores (1.0 – 10.0)
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <div className="flex justify-between text-xs font-bold mb-1 tabular-nums">
                            <span>Narrative Continuity</span>
                            <span className="font-mono">
                              {selectedTurn.narrative_continuity_score.toFixed(1)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="10"
                            step="0.1"
                            value={selectedTurn.narrative_continuity_score}
                            onChange={(e) =>
                              handleUpdateSelectedTurn({
                                narrative_continuity_score: parseFloat(
                                  e.target.value
                                ),
                              })
                            }
                            className="w-full accent-[#FF5CA8]"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between text-xs font-bold mb-1 tabular-nums">
                            <span>Character Consistency</span>
                            <span className="font-mono">
                              {selectedTurn.character_consistency_score.toFixed(1)}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="10"
                            step="0.1"
                            value={selectedTurn.character_consistency_score}
                            onChange={(e) =>
                              handleUpdateSelectedTurn({
                                character_consistency_score: parseFloat(
                                  e.target.value
                                ),
                              })
                            }
                            className="w-full accent-[#7C3AED]"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between text-xs font-bold mb-1 tabular-nums">
                            <span>Emotional Expressiveness</span>
                            <span className="font-mono">
                              {selectedTurn.emotional_expressiveness_score.toFixed(
                                1
                              )}
                            </span>
                          </div>
                          <input
                            type="range"
                            min="1"
                            max="10"
                            step="0.1"
                            value={selectedTurn.emotional_expressiveness_score}
                            onChange={(e) =>
                              handleUpdateSelectedTurn({
                                emotional_expressiveness_score: parseFloat(
                                  e.target.value
                                ),
                              })
                            }
                            className="w-full accent-[#FF5CA8]"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-xs font-bold text-[#2A114B] mb-1">
                          Human Analyst Evaluation Notes
                        </label>
                        <textarea
                          rows={2}
                          value={selectedTurn.human_evaluation_notes}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              human_evaluation_notes: e.target.value,
                            })
                          }
                          className="w-full p-2.5 rounded-xl border-2 border-[#2A114B] bg-white text-xs text-[#2A114B]"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-[#FFF8EC] p-8 rounded-2xl border-[2.5px] border-[#2A114B] text-center">
                  <p className="font-display text-base font-bold text-[#2A114B]">
                    No turns match the current filter.
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        /* SUB-TAB 2: Guided SFT & DPO Pair Studio + Schema-Versioned Export / Import */
        <div className="space-y-6">
          {/* Guided Beginner-Friendly Explainer Banner */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-[#FFF8EC] p-4 rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B]">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="px-2.5 py-0.5 rounded-full border-2 border-[#2A114B] bg-[#DFC5FE] font-display text-xs font-bold">
                  Stage 1: SFT (Supervised Fine-Tuning)
                </span>
                <span className="text-xs font-mono text-[#5C3B88]">
                  Imitation &amp; Voice Alignment
                </span>
              </div>
              <p className="text-xs text-[#2A114B] leading-relaxed">
                Compiles your highest-scoring <strong>Gemini 2.5 Pro</strong>,{' '}
                <strong>Gemini 3.1 Pro</strong>, and <strong>Gemini 3 Flash</strong>{' '}
                threads into multi-turn examples formatted for{' '}
                <code className="font-mono font-bold">
                  Tinman-Lab/Tinman-gemma4-companion-merged
                </code>{' '}
                (<code className="font-mono">&lt;|turn&gt;user...&lt;turn|&gt;</code>).
              </p>
            </div>

            <div className="bg-[#FFF8EC] p-4 rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B]">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="px-2.5 py-0.5 rounded-full border-2 border-[#2A114B] bg-[#FF85C0] text-white font-display text-xs font-bold">
                  Stage 2: DPO (Direct Preference Optimization)
                </span>
                <span className="text-xs font-mono text-[#5C3B88]">
                  Chosen vs. Rejected Contrast
                </span>
              </div>
              <p className="text-xs text-[#2A114B] leading-relaxed">
                Pairs your favorite <strong>Chosen</strong> response (e.g. from{' '}
                <strong>Pro 2.5</strong> or <strong>Pro 3.1</strong>) next to a{' '}
                <strong>Rejected</strong> response (e.g. from{' '}
                <strong>Flash 3</strong> or 1-Click Teacher Contrast Drafter) so Tinman
                learns what tone and habits to avoid.
              </p>
            </div>
          </div>

          {/* Side-by-Side DPO Chosen vs Rejected Pair Builder */}
          {selectedTurn && (
            <div className="bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_5px_0_#2A114B] overflow-hidden">
              <div className="px-5 py-3.5 bg-[#DFC5FE] border-b-2 border-[#2A114B] flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <GitCompare className="w-4 h-4 text-[#2A114B]" />
                  <span className="font-display font-bold text-sm text-[#2A114B]">
                    DPO Preference Pair Builder — {selectedTurn.conversation_title} (Turn #
                    {selectedTurn.turn_number})
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    value={selectedTurn.Id}
                    onChange={(e) => setSelectedTurnId(e.target.value)}
                    className="px-3 py-1.5 rounded-full border-2 border-[#2A114B] bg-white text-xs font-mono font-bold"
                  >
                    {state.outputs.map((o) => (
                      <option key={o.id} value={o.id}>
                        [{getModelBadgeMeta(o.model_version).shortBadge}] {o.conversation_id} #
                        {o.turn_number}
                      </option>
                    ))}
                  </select>
                  <button
                    disabled={isDraftingContrast}
                    onClick={() =>
                      handleDraftDpoContrast(
                        selectedTurn,
                        selectedTurn.rejection_reason || 'robotic_tone'
                      )
                    }
                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border-2 border-[#2A114B] bg-[#FF5CA8] text-white font-display text-xs font-bold shadow-[0_2px_0_#2A114B] cursor-pointer disabled:opacity-50"
                  >
                    <Wand2 className="w-3.5 h-3.5" />
                    {isDraftingContrast
                      ? 'Drafting Contrast...'
                      : '1-Click Auto-Draft Rejected Contrast'}
                  </button>
                </div>
              </div>

              <div className="p-5 space-y-4">
                <div className="p-3 rounded-xl border-2 border-[#2A114B] bg-[#F3E8FF]/60 text-xs">
                  <span className="font-bold text-[#5C3B88] uppercase font-mono block mb-1">
                    Shared Prompt Context:
                  </span>
                  <p className="font-bold text-[#2A114B] text-sm">
                    {selectedTurn.user_prompt}
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Chosen Column */}
                  <div className="p-4 rounded-2xl border-2 border-[#2A114B] bg-[#A7F3D0]/30 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="px-2.5 py-0.5 rounded-full border-2 border-[#2A114B] bg-[#A7F3D0] text-[#065F46] font-display text-xs font-bold">
                        ✓ CHOSEN RESPONSE (Target Behavior)
                      </span>
                      <span className="text-xs font-mono font-bold text-[#2A114B]">
                        Source: {getModelBadgeMeta(selectedTurn.model_version).displayName}
                      </span>
                    </div>
                    <textarea
                      rows={5}
                      value={selectedTurn.model_output}
                      onChange={(e) =>
                        handleUpdateSelectedTurn({ model_output: e.target.value })
                      }
                      className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-xs leading-relaxed text-[#2A114B]"
                    />
                  </div>

                  {/* Rejected Column */}
                  <div className="p-4 rounded-2xl border-2 border-[#2A114B] bg-[#FECDD3]/35 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="px-2.5 py-0.5 rounded-full border-2 border-[#2A114B] bg-[#FECDD3] text-[#881337] font-display text-xs font-bold">
                        ✕ REJECTED RESPONSE (Negative Contrast)
                      </span>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="text"
                          value={selectedTurn.rejected_model_version}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              rejected_model_version: e.target.value,
                            })
                          }
                          placeholder="e.g. gemini-3-flash"
                          className="px-2 py-0.5 rounded-lg border border-[#2A114B] bg-white text-[11px] font-mono w-36"
                        />
                        <select
                          value={selectedTurn.rejection_reason || 'robotic_tone'}
                          onChange={(e) =>
                            handleUpdateSelectedTurn({
                              rejection_reason: e.target.value,
                            })
                          }
                          className="px-2 py-0.5 rounded-lg border border-[#2A114B] bg-white text-[11px] font-mono"
                        >
                          <option value="robotic_tone">robotic_tone</option>
                          <option value="too_brief">too_brief</option>
                          <option value="out_of_character">out_of_character</option>
                          <option value="continuity_break">continuity_break</option>
                        </select>
                      </div>
                    </div>
                    <textarea
                      rows={5}
                      value={selectedTurn.rejected_output}
                      onChange={(e) =>
                        handleUpdateSelectedTurn({
                          rejected_output: e.target.value,
                        })
                      }
                      placeholder="Paste a weaker response (e.g. from Flash 3) or click '1-Click Auto-Draft Rejected Contrast' above..."
                      className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-xs leading-relaxed text-[#2A114B]"
                    />
                  </div>
                </div>

                {/* Live Gemma 4 Control-Token Serialization Preview */}
                <div className="p-3.5 rounded-2xl border-2 border-[#2A114B] bg-[#26104B] text-[#F3E8FF]">
                  <div className="flex items-center justify-between mb-1.5 text-xs">
                    <span className="font-display font-bold text-[#FACC15]">
                      Verified Gemma 4 E4B Turn Serialization Preview (
                      <code>Tinman-Lab/Tinman-gemma4-companion-merged</code>)
                    </span>
                    <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useThinkTokenInExport}
                        onChange={(e) => setUseThinkTokenInExport(e.target.checked)}
                      />
                      <span>Include Verified &lt;|think|&gt; Control Prefix</span>
                    </label>
                  </div>
                  <pre className="text-[11px] font-mono whitespace-pre-wrap overflow-x-auto leading-relaxed text-[#E9D5FF]">
                    {formatGemma4TurnString(
                      selectedTurn.system_prompt,
                      selectedTurn.user_prompt,
                      selectedTurn.model_output,
                      useThinkTokenInExport
                    )}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {/* Export Compiler & Schema v1.2.0 Import Validator */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B] p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-display text-base font-bold text-[#2A114B] flex items-center gap-2">
                  <FileCode2 className="w-5 h-5 text-[#7C3AED]" />
                  Leak-Guarded JSON / JSONL Export Compiler (Schema v1.2.0)
                </h3>
                <button
                  onClick={handleDownloadExport}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-full border-2 border-[#2A114B] bg-[#FF5CA8] text-white font-display text-xs font-bold shadow-[0_3px_0_#2A114B] cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  Download {exportFormat.toUpperCase()}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                    Dataset Export Format
                  </label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as any)}
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-bold text-[#2A114B]"
                  >
                    <option value="sft_jsonl">
                      Supervised Fine-Tuning (SFT Multi-Turn JSONL)
                    </option>
                    <option value="dpo_jsonl">
                      Preference Dataset (DPO Chosen/Rejected JSONL)
                    </option>
                    <option value="eval_jsonl">
                      Behavioral Evaluation Corpus (Annotated JSONL)
                    </option>
                    <option value="full_json">
                      Full SQLite Workbench Snapshot (Schema v1.2.0 JSON)
                    </option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-bold text-[#5C3B88] mb-1">
                    Filter Export by Gemini Teacher Model
                  </label>
                  <select
                    value={exportModelFilter}
                    onChange={(e) => setExportModelFilter(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono font-bold text-[#2A114B]"
                  >
                    <option value="ALL">All Eligible Gemini Models Combined</option>
                    {allKnownModelIds.map((mId) => (
                      <option key={mId} value={mId}>
                        Only {getModelBadgeMeta(mId).displayName} ({mId})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Freeze into Stage 3 Lineage Snapshot */}
              <div className="p-3.5 rounded-xl border-2 border-[#2A114B] bg-[#F3E8FF] flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[#2A114B]">
                    Snapshot Tag:
                  </span>
                  <input
                    type="text"
                    value={snapshotTag}
                    onChange={(e) => setSnapshotTag(e.target.value)}
                    className="px-2.5 py-1 rounded-lg border-2 border-[#2A114B] bg-white text-xs font-mono font-bold w-44"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCreateDatasetSnapshot('sft')}
                    className="px-3 py-1.5 rounded-full border-2 border-[#2A114B] bg-[#FACC15] text-xs font-display font-bold text-[#2A114B] shadow-[0_2px_0_#2A114B] cursor-pointer"
                  >
                    + Register SFT Snapshot in Lineage
                  </button>
                  <button
                    onClick={() => handleCreateDatasetSnapshot('dpo')}
                    className="px-3 py-1.5 rounded-full border-2 border-[#2A114B] bg-[#DFC5FE] text-xs font-display font-bold text-[#2A114B] shadow-[0_2px_0_#2A114B] cursor-pointer"
                  >
                    + Register DPO Snapshot in Lineage
                  </button>
                </div>
              </div>

              <div>
                <div className="text-xs font-bold text-[#5C3B88] mb-1">
                  Live Schema-Versioned Output Preview:
                </div>
                <pre className="w-full h-64 overflow-auto p-3.5 rounded-xl border-2 border-[#2A114B] bg-[#26104B] text-[#E9D5FF] text-[11px] font-mono leading-relaxed">
                  {compiledExportText || '// No eligible turns match current filter'}
                </pre>
              </div>
            </div>

            {/* Right Column: Schema Validator & Migration Importer */}
            <div className="lg:col-span-5 bg-[#FFF8EC] rounded-2xl border-[2.5px] border-[#2A114B] shadow-[0_4px_0_#2A114B] p-5 space-y-4">
              <h3 className="font-display text-base font-bold text-[#2A114B] flex items-center gap-2">
                <Upload className="w-5 h-5 text-[#FF5CA8]" />
                Schema Migration &amp; Dry-Run Import Validator
              </h3>
              <p className="text-xs text-[#5C3B88]">
                Paste or upload JSON / JSONL transcripts from OpenRouter or AI Studio.
                Validates required fields, preserves source Gemini model attribution,
                and runs the 4-Layer Leakage Firewall before SQLite insertion.
              </p>

              <textarea
                rows={6}
                value={importRawText}
                onChange={(e) => setImportRawText(e.target.value)}
                placeholder='{"model_version": "google/gemini-2.5-pro", "user_prompt": "...", "model_output": "..."}'
                className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono text-[#2A114B]"
              />

              <div className="flex items-center gap-2">
                <button
                  onClick={handleValidateImport}
                  className="px-4 py-2 rounded-full border-2 border-[#2A114B] bg-[#DFC5FE] hover:bg-[#C084FC] text-[#2A114B] font-display text-xs font-bold shadow-[0_2px_0_#2A114B] cursor-pointer"
                >
                  Run Dry-Run Schema &amp; Leakage Validation
                </button>
                {importValidationResult && (
                  <button
                    onClick={handleCommitImport}
                    className="px-4 py-2 rounded-full border-2 border-[#2A114B] bg-[#A7F3D0] text-[#065F46] font-display text-xs font-bold shadow-[0_2px_0_#2A114B] cursor-pointer"
                  >
                    Commit {importValidationResult.valid_count} Valid Turns to SQLite
                  </button>
                )}
              </div>

              {importValidationResult && (
                <div className="p-3.5 rounded-xl border-2 border-[#2A114B] bg-[#F3E8FF] space-y-2 text-xs">
                  <div className="font-bold text-[#2A114B]">
                    Detected Schema: {importValidationResult.detected_schema_version}
                  </div>
                  <div className="font-mono tabular-nums">
                    Total Rows: {importValidationResult.total_records} · Valid:{' '}
                    {importValidationResult.valid_count} · Leakage Flagged:{' '}
                    {importValidationResult.leakage_flagged_count}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Modal to Log New Turn / Multi-Turn Output */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 bg-[#26104B]/80 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-[#FFF8EC] w-full max-w-3xl rounded-3xl border-[3px] border-[#2A114B] shadow-[0_8px_0_#2A114B] overflow-hidden my-8">
            <div className="px-6 py-4 bg-gradient-to-r from-[#7C3AED] to-[#FF5CA8] border-b-[3px] border-[#2A114B] flex items-center justify-between text-white">
              <h3 className="font-display text-lg font-bold flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Log Gemini Output Turn to SQLite (Manual Paste or Live Teacher)
              </h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="px-3 py-1 rounded-full border-2 border-[#2A114B] bg-[#FFF8EC] text-[#2A114B] font-display text-xs font-bold"
              >
                Close
              </button>
            </div>

            <form onSubmit={handleCreateTurn} className="p-6 space-y-4 max-h-[80vh] overflow-y-auto">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Select Gemini Model Version
                  </label>
                  <select
                    value={newTurnForm.model_version}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        model_version: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono font-bold text-[#2A114B]"
                  >
                    {GEMINI_TEACHER_MODELS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.displayName} ({m.id})
                      </option>
                    ))}
                    <option value="CUSTOM">+ Custom Gemini Model String...</option>
                  </select>
                </div>

                {newTurnForm.model_version === 'CUSTOM' ? (
                  <div>
                    <label className="block text-xs font-bold text-[#2A114B] mb-1">
                      Custom Model Identifier
                    </label>
                    <input
                      type="text"
                      required
                      value={newTurnForm.custom_model_version}
                      onChange={(e) =>
                        setNewTurnForm({
                          ...newTurnForm,
                          custom_model_version: e.target.value,
                        })
                      }
                      placeholder="e.g. google/gemini-2.5-pro-exp"
                      className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono"
                    />
                  </div>
                ) : (
                  <div>
                    <label className="block text-xs font-bold text-[#2A114B] mb-1">
                      Provider / Source
                    </label>
                    <select
                      value={newTurnForm.provider_source}
                      onChange={(e) =>
                        setNewTurnForm({
                          ...newTurnForm,
                          provider_source: e.target.value,
                        })
                      }
                      className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-bold"
                    >
                      <option value="OpenRouter">OpenRouter</option>
                      <option value="AI Studio">AI Studio</option>
                      <option value="Live API">Live API</option>
                      <option value="Imported">Imported</option>
                    </select>
                  </div>
                )}

                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Output Date &amp; Time
                  </label>
                  <input
                    type="datetime-local"
                    value={newTurnForm.recorded_at}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        recorded_at: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Conversation ID
                  </label>
                  <input
                    type="text"
                    required
                    value={newTurnForm.conversation_id}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        conversation_id: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Thread Title
                  </label>
                  <input
                    type="text"
                    required
                    value={newTurnForm.conversation_title}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        conversation_title: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Turn Number
                  </label>
                  <input
                    type="number"
                    min={1}
                    required
                    value={newTurnForm.turn_number}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        turn_number: Number(e.target.value),
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-bold text-[#2A114B] mb-1">
                  System Prompt
                </label>
                <textarea
                  rows={2}
                  value={newTurnForm.system_prompt}
                  onChange={(e) =>
                    setNewTurnForm({
                      ...newTurnForm,
                      system_prompt: e.target.value,
                    })
                  }
                  className="w-full p-2.5 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-bold text-[#2A114B]">
                    User Prompt
                  </label>
                  <button
                    type="button"
                    disabled={isGeneratingLive}
                    onClick={handleLiveTeacherGenerate}
                    className="px-3 py-1 rounded-full border-2 border-[#2A114B] bg-[#DFC5FE] hover:bg-[#C084FC] text-[#2A114B] font-display text-xs font-bold cursor-pointer"
                  >
                    {isGeneratingLive
                      ? 'Generating via Live Gemini...'
                      : '⚡ Generate Output via Live Gemini API'}
                  </button>
                </div>
                <textarea
                  rows={2}
                  required
                  value={newTurnForm.user_prompt}
                  onChange={(e) =>
                    setNewTurnForm({
                      ...newTurnForm,
                      user_prompt: e.target.value,
                    })
                  }
                  placeholder="Enter user message..."
                  className="w-full p-2.5 rounded-xl border-2 border-[#2A114B] bg-white text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-[#2A114B] mb-1">
                  Gemini Model Output (Automatically preserved in immutable original_transcript)
                </label>
                <textarea
                  rows={4}
                  required
                  value={newTurnForm.model_output}
                  onChange={(e) =>
                    setNewTurnForm({
                      ...newTurnForm,
                      model_output: e.target.value,
                    })
                  }
                  placeholder="Paste your Gemini 2.5 Pro, 3.1 Pro, or 3 Flash output here (or click Generate above)..."
                  className="w-full p-3 rounded-xl border-2 border-[#2A114B] bg-white text-sm"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Persona Archetype
                  </label>
                  <input
                    type="text"
                    value={newTurnForm.persona_archetype}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        persona_archetype: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Behavioral Category
                  </label>
                  <input
                    type="text"
                    value={newTurnForm.behavioral_category}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        behavioral_category: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-[#2A114B] mb-1">
                    Tags (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={newTurnForm.tags_input}
                    onChange={(e) =>
                      setNewTurnForm({
                        ...newTurnForm,
                        tags_input: e.target.value,
                      })
                    }
                    className="w-full px-3 py-2 rounded-xl border-2 border-[#2A114B] bg-white text-xs"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-3 border-t-2 border-[#2A114B]/20">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 rounded-full border-2 border-[#2A114B] bg-white font-display text-xs font-bold"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-full border-2 border-[#2A114B] bg-[#FF5CA8] text-white font-display text-sm font-bold shadow-[0_3px_0_#2A114B] cursor-pointer"
                >
                  Save Turn &amp; Run Leakage Check
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
