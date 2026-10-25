"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { API, authHeaders } from "@/lib/api";

const inputStyle =
  "w-full rounded-xl border border-violet-300/20 bg-[#100b19] px-3.5 py-2.5 text-sm text-violet-50 outline-none focus:border-violet-400 transition";

type Audit = {
  id: string;
  name: string;
  status: string;
  facts?: any[];
  questions?: any[];
  target?: any;
  protocol?: any;
  phases?: any;
  reference_target?: any;
  counterfactual_target?: any;
  track_metadata?: any;
};

const PROVIDERS = [
  {
    id: "gemini",
    name: "Google Gemini",
    defaultModel: "gemini-3.5-flash-lite",
    envVar: "GEMINI_API_KEY",
    desc: "Google Gemini Interactions / generateContent API",
  },
  {
    id: "openai_compatible",
    name: "OpenAI / Compatible",
    defaultModel: "gpt-4o-mini",
    envVar: "OPENAI_API_KEY",
    desc: "OpenAI, Groq, Together, DeepSeek, vLLM",
  },
  {
    id: "anthropic",
    name: "Anthropic Claude",
    defaultModel: "claude-3-5-sonnet-20241022",
    envVar: "ANTHROPIC_API_KEY",
    desc: "Anthropic Messages API",
  },
  {
    id: "ollama",
    name: "Local Ollama",
    defaultModel: "llama3.2",
    envVar: "",
    desc: "Local private open-weight models (localhost:11434)",
  },
  {
    id: "guided",
    name: "Guided / Black-Box Web Chat",
    defaultModel: "Web Chatbot",
    envVar: "",
    desc: "Manual entry for any consumer chatbot (ChatGPT, Claude, Copilot)",
  },
];

const OPERATION_CATEGORIES = [
  {
    id: "blackbox_disclosure_test",
    label: "Black-Box Disclosure Testing",
    desc: "Probe hosted model without modification of underlying weights.",
  },
  {
    id: "model_weight_unlearning",
    label: "Model Weight Unlearning",
    desc: "Model parameters modified by unlearning procedure (e.g. LoRA, GA, SCRUB).",
  },
  {
    id: "external_memory_deletion",
    label: "External Memory / RAG Deletion",
    desc: "Target information purged from vector store or persistent memory layer.",
  },
  {
    id: "context_purge_reset",
    label: "Context Window Reset / Purge",
    desc: "Conversation reset / prompt purge (weights remain unchanged).",
  },
  {
    id: "guided_manual",
    label: "Guided / Manual Review",
    desc: "Human-in-the-loop manual testing across independent sessions.",
  },
];

export default function ForgettingAuditPage() {
  const [audit, setAudit] = useState<Audit | null>(null);
  const [name, setName] = useState("Controlled AI Forgetting Audit");
  const [auditTrack, setAuditTrack] = useState<"track_a_model_unlearning" | "track_b_memory_deletion">("track_a_model_unlearning");
  
  // Track A parameters
  const [unlearningMethod, setUnlearningMethod] = useState("Gradient Ascent (GA) with SCRUB regularization");
  const [preCheckpoint, setPreCheckpoint] = useState("base-model-v1");
  const [postCheckpoint, setPostCheckpoint] = useState("unlearned-model-v1");

  // Track B parameters
  const [storageComponent, setStorageComponent] = useState("vector_database");
  const [storageVerificationStatus, setStorageVerificationStatus] = useState("verified_deleted");

  const [target, setTarget] = useState("gemini");
  const [model, setModel] = useState("gemini-3.5-flash-lite");
  const [baseUrl, setBaseUrl] = useState("https://generativelanguage.googleapis.com");
  const [apiKeyEnv, setApiKeyEnv] = useState("GEMINI_API_KEY");
  const [operationCategory, setOperationCategory] = useState("model_weight_unlearning");
  const [tier, setTier] = useState("production");

  // Paired control reference (Group B)
  const [referenceType, setReferenceType] = useState("gemini");
  const [referenceModel, setReferenceModel] = useState("gemini-3.5-flash-lite");
  const [referenceBaseUrl, setReferenceBaseUrl] = useState("https://generativelanguage.googleapis.com");
  const [referenceApiKeyEnv, setReferenceApiKeyEnv] = useState("GEMINI_API_KEY");
  const [referenceTier, setReferenceTier] = useState("never-exposed control");

  // Counterfactual control (Group C)
  const [counterfactualType, setCounterfactualType] = useState("gemini");
  const [counterfactualModel, setCounterfactualModel] = useState("gemini-3.5-flash-lite");
  const [counterfactualBaseUrl, setCounterfactualBaseUrl] = useState("https://generativelanguage.googleapis.com");
  const [counterfactualApiKeyEnv, setCounterfactualApiKeyEnv] = useState("GEMINI_API_KEY");

  const [method, setMethod] = useState("conversation");
  const [forgetNotes, setForgetNotes] = useState(
    "Documented operation: Verified unlearning / deletion procedure executed on target system."
  );

  // Fact configuration
  const [fact, setFact] = useState("Who wrote Romeo and Juliet?");
  const [answer, setAnswer] = useState("William Shakespeare");
  const [targetQuestion, setTargetQuestion] = useState("Who wrote Romeo and Juliet?");
  const [retainedQ, setRetainedQ] = useState("What is the capital city of France?");
  const [retainedA, setRetainedA] = useState("Paris");

  // Dataset filtering
  const [filterStart, setFilterStart] = useState(0);
  const [filterLimit, setFilterLimit] = useState(10);

  // Custom probe
  const [question, setQuestion] = useState("");
  const [questionFact, setQuestionFact] = useState("");
  const [phase, setPhase] = useState("baseline");
  const [manualQ, setManualQ] = useState("");
  const [manualAnswer, setManualAnswer] = useState("");

  const [report, setReport] = useState<any>(null);
  const [exposureResults, setExposureResults] = useState<any[]>([]);
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [copiedLatex, setCopiedLatex] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const aid = params.get("audit_id");
    const trk = params.get("track");
    if (trk === "track_a_model_unlearning" || trk === "track_b_memory_deletion") {
      setAuditTrack(trk);
    }
    if (aid) {
      void act(async () => {
        const fresh = await req(`/forgetting-audits/${aid}`);
        setAudit(fresh);
        if (fresh?.track_metadata?.track) {
          setAuditTrack(fresh.track_metadata.track);
        }
      });
    }
  }, []);

  function handleProviderChange(providerId: string) {
    setTarget(providerId);
    const p = PROVIDERS.find((x) => x.id === providerId);
    if (p) {
      setModel(p.defaultModel);
      setApiKeyEnv(p.envVar);
      if (providerId === "ollama") setBaseUrl("http://localhost:11434");
      else if (providerId === "openai_compatible") setBaseUrl("https://api.openai.com/v1");
      else if (providerId === "anthropic") setBaseUrl("https://api.anthropic.com/v1");
      else if (providerId === "gemini") setBaseUrl("https://generativelanguage.googleapis.com");
    }
  }

  async function req(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers || {});
    const auth = authHeaders();
    Object.entries(auth).forEach(([k, v]) => headers.set(k, v));
    const r = await fetch(API + path, { ...init, headers, cache: "no-store" });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
    return j;
  }

  async function act(fn: () => Promise<any>) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await fn();
      setMessage(typeof result?.message === "string" ? result.message : "Done.");
      if (audit?.id) {
        const fresh = await req(`/forgetting-audits/${audit.id}`);
        setAudit(fresh);
      }
      return result;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function create() {
    const j = await act(() =>
      req("/forgetting-audits", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          audit_track: auditTrack,
          unlearning_method: unlearningMethod,
          pre_unlearn_checkpoint: preCheckpoint,
          post_unlearn_checkpoint: postCheckpoint,
          storage_component: storageComponent,
          storage_verification_status: storageVerificationStatus,
          target_type: target,
          model,
          base_url: baseUrl,
          api_key_env: apiKeyEnv,
          tier_label: tier,
          operation_category: auditTrack === "track_b_memory_deletion" ? "external_memory_deletion" : operationCategory,
          reference_target_type: referenceType,
          reference_model: referenceModel,
          reference_base_url: referenceBaseUrl,
          reference_api_key_env: referenceApiKeyEnv,
          reference_tier_label: referenceTier,
          counterfactual_target_type: counterfactualType,
          counterfactual_model: counterfactualModel,
          counterfactual_base_url: counterfactualBaseUrl,
          counterfactual_api_key_env: counterfactualApiKeyEnv,
          counterfactual_tier_label: "unmitigated control",
          exposure_method: method,
          forget_operation: forgetNotes,
          questions_per_fact: 5,
          repetitions: 1,
        }),
      })
    );
    if (j?.id) {
      const fresh = await req(`/forgetting-audits/${j.id}`);
      setAudit(fresh);
    }
  }

  async function upload(file?: File) {
    if (!file || !audit) return;
    await act(async () => {
      const fd = new FormData();
      fd.append("file", file);
      return req(`/forgetting-audits/${audit.id}/dataset`, { method: "POST", body: fd });
    });
  }

  async function filterDataset() {
    if (!audit) return;
    await act(() =>
      req(`/forgetting-audits/${audit.id}/dataset/filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_index: filterStart, limit: filterLimit }),
      })
    );
  }

  async function addFact() {
    if (!audit) return;
    const j = await act(() =>
      req(`/forgetting-audits/${audit.id}/facts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          facts: [
            {
              fact,
              expected_answer: answer,
              target_question: targetQuestion,
              retained_question: retainedQ,
              retained_answer: retainedA,
            },
          ],
        }),
      })
    );
    if (j) {
      setFact("");
      setAnswer("");
      setTargetQuestion("What is the exact target information?");
      setRetainedQ("");
      setRetainedA("");
    }
  }

  async function generate() {
    if (audit) await act(() => req(`/forgetting-audits/${audit.id}/generate-probes`, { method: "POST" }));
  }

  async function addQuestion() {
    if (!audit) return;
    const j = await act(() =>
      req(`/forgetting-audits/${audit.id}/questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fact_id: questionFact, question, kind: "custom" }),
      })
    );
    if (j) setQuestion("");
  }

  async function editQuestion(q: any) {
    if (!audit) return;
    await act(() =>
      req(`/forgetting-audits/${audit.id}/questions/${q.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: editValues[q.id] ?? q.question }),
      })
    );
  }

  async function run(p: string) {
    if (!audit) return;
    await act(() => req(`/forgetting-audits/${audit.id}/run/${p}`, { method: "POST" }));
    setPhase(p);
  }

  async function expose() {
    if (!audit) return;
    const j = await act(() => req(`/forgetting-audits/${audit.id}/expose`, { method: "POST" }));
    if (j?.results) setExposureResults(j.results);
  }

  async function mark() {
    if (audit)
      await act(() =>
        req(`/forgetting-audits/${audit.id}/mark-forgotten`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            operation_notes: forgetNotes,
            storage_verification_status: storageVerificationStatus,
          }),
        })
      );
  }

  async function manualSubmit() {
    if (!audit || !manualQ) return;
    const j = await act(() =>
      req(`/forgetting-audits/${audit.id}/manual-response`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question_id: manualQ, phase, response: manualAnswer }),
      })
    );
    if (j) setManualAnswer("");
  }

  async function loadReport() {
    if (!audit) return;
    const j = await act(() => req(`/forgetting-audits/${audit.id}/report`));
    if (j) setReport(j);
  }

  async function downloadExport(type: "json" | "csv" | "latex") {
    if (!audit) return;
    try {
      const headers = new Headers();
      const auth = authHeaders();
      Object.entries(auth).forEach(([k, v]) => headers.set(k, v));
      const res = await fetch(`${API}/forgetting-audits/${audit.id}/export/${type}`, {
        headers,
      });
      if (!res.ok) throw new Error(`Export failed (${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const ext = type === "latex" ? "tex" : type;
      a.download = `rena_audit_${audit.id.slice(0, 8)}.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function copyLatexTable() {
    if (!report) return;
    const tableText = report.latex_table || "";
    navigator.clipboard.writeText(tableText);
    setCopiedLatex(true);
    setTimeout(() => setCopiedLatex(false), 3000);
  }

  const questions = audit?.questions || [];
  const facts = audit?.facts || [];

  return (
    <main className="min-h-screen bg-[#090610] text-white">
      <header className="border-b border-violet-300/10 px-5 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <Link href="/dashboard" className="flex items-center gap-2 font-bold tracking-widest text-violet-100">
            <span className="rounded-lg bg-violet-600 px-2 py-0.5 text-xs text-white">RENA</span>
            <span>FORGETTING AUDIT STUDIO</span>
          </Link>
          <div className="flex items-center gap-3">
            <Link href="/dashboard" className="text-xs text-violet-300 hover:text-white">
              ← Back to Overview
            </Link>
            <Link href="/llm" className="text-xs text-violet-300 hover:text-white">
              LLM Runner ↗
            </Link>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300/30 bg-amber-400/10 px-3 py-1 text-xs font-semibold text-amber-200">
            TWO-TRACK SCIENTIFIC AUDIT FRAMEWORK
          </span>
          <h1 className="mt-3 text-3xl font-bold sm:text-4xl">
            Reliable Evaluation for Neural Unlearning Audits<span className="text-amber-300">.</span>
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-violet-100/75">
            Test whether designated target information remains accessible after a documented forgetting operation.
            Evaluate live models with controlled reference comparisons (Groups A, B, C), multi-angle probing (direct, paraphrase, indirect, multi-turn),
            Wilson 95% confidence intervals, and causal attribution without conflating model weights and application storage.
          </p>
        </div>

        {error && <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-200">{error}</div>}
        {message && <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/5 p-4 text-sm text-emerald-200">{message}</div>}

        {/* STEP 1: TARGET MODEL & TRACK CONFIGURATION */}
        {!audit ? (
          <section className="rounded-2xl border border-violet-300/15 bg-white/[.03] p-6 shadow-xl">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-amber-300">Step 1 of 6</span>
                <h2 className="text-xl font-semibold">Select Evaluation Track & AI Architecture</h2>
              </div>
            </div>

            {/* TRACK SELECTOR TILES */}
            <div className="mt-5 grid gap-4 sm:grid-cols-2">
              <div
                onClick={() => {
                  setAuditTrack("track_a_model_unlearning");
                  setOperationCategory("model_weight_unlearning");
                }}
                className={`cursor-pointer rounded-xl border p-4 transition ${
                  auditTrack === "track_a_model_unlearning"
                    ? "border-violet-400 bg-violet-600/15 shadow-lg shadow-violet-950/40"
                    : "border-violet-300/15 bg-black/40 hover:border-violet-300/30"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-violet-200">Track A</span>
                  <span className="rounded bg-violet-600/30 px-2 py-0.5 text-[10px] font-semibold text-violet-200">Parameter Layer</span>
                </div>
                <h3 className="mt-2 text-base font-bold text-white">Model-Level Weight Unlearning</h3>
                <p className="mt-1 text-xs text-violet-100/70">
                  Audit checkpoints or models subjected to parameter modification algorithms (e.g. Gradient Ascent, SCRUB, NPO, LoRA fine-tuning).
                </p>
              </div>

              <div
                onClick={() => {
                  setAuditTrack("track_b_memory_deletion");
                  setOperationCategory("external_memory_deletion");
                }}
                className={`cursor-pointer rounded-xl border p-4 transition ${
                  auditTrack === "track_b_memory_deletion"
                    ? "border-amber-400 bg-amber-400/10 shadow-lg shadow-amber-950/40"
                    : "border-violet-300/15 bg-black/40 hover:border-violet-300/30"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-300">Track B</span>
                  <span className="rounded bg-amber-400/20 px-2 py-0.5 text-[10px] font-semibold text-amber-200">Application Storage</span>
                </div>
                <h3 className="mt-2 text-base font-bold text-white">Application Memory & Retrieval Deletion</h3>
                <p className="mt-1 text-xs text-violet-100/70">
                  Audit data purged from vector stores, RAG indices, conversation memory buffers, or prompt caches without claiming weight modification.
                </p>
              </div>
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <label className="text-xs text-violet-200">
                Audit / Experiment Name
                <input className={`${inputStyle} mt-2`} value={name} onChange={(e) => setName(e.target.value)} />
              </label>

              <label className="text-xs text-violet-200">
                Model Provider / Architecture
                <select className={`${inputStyle} mt-2`} value={target} onChange={(e) => handleProviderChange(e.target.value)}>
                  {PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="text-xs text-violet-200">
                Model Identifier
                <input
                  className={`${inputStyle} mt-2`}
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="e.g. gemini-3.5-flash-lite"
                />
              </label>

              {/* TRACK-SPECIFIC INPUTS */}
              {auditTrack === "track_a_model_unlearning" ? (
                <>
                  <label className="text-xs text-violet-200">
                    Unlearning Method
                    <input
                      className={`${inputStyle} mt-2`}
                      value={unlearningMethod}
                      onChange={(e) => setUnlearningMethod(e.target.value)}
                      placeholder="e.g. Gradient Ascent (GA), SCRUB, NPO"
                    />
                  </label>
                  <label className="text-xs text-violet-200">
                    Pre-Unlearn Model Checkpoint
                    <input
                      className={`${inputStyle} mt-2`}
                      value={preCheckpoint}
                      onChange={(e) => setPreCheckpoint(e.target.value)}
                      placeholder="e.g. llama3.2-base"
                    />
                  </label>
                  <label className="text-xs text-violet-200">
                    Post-Unlearn Model Checkpoint
                    <input
                      className={`${inputStyle} mt-2`}
                      value={postCheckpoint}
                      onChange={(e) => setPostCheckpoint(e.target.value)}
                      placeholder="e.g. llama3.2-unlearned-ckpt-v2"
                    />
                  </label>
                </>
              ) : (
                <>
                  <label className="text-xs text-violet-200">
                    Target Storage Component
                    <select
                      className={`${inputStyle} mt-2`}
                      value={storageComponent}
                      onChange={(e) => setStorageComponent(e.target.value)}
                    >
                      <option value="vector_database">Vector Database (Milvus / Chroma / Pinecone)</option>
                      <option value="rag_document_index">RAG Document Index</option>
                      <option value="memory_store">Persistent Memory Store (Redis / SQL)</option>
                      <option value="session_context">Conversation Session Context</option>
                    </select>
                  </label>
                  <label className="text-xs text-violet-200">
                    Store Deletion Verification
                    <select
                      className={`${inputStyle} mt-2`}
                      value={storageVerificationStatus}
                      onChange={(e) => setStorageVerificationStatus(e.target.value)}
                    >
                      <option value="verified_deleted">Verified Deleted at Source of Truth</option>
                      <option value="record_soft_deleted">Soft-Deleted / Flagged Invisible</option>
                      <option value="store_unmodified">Store Unmodified (Control Test)</option>
                      <option value="unverified">Unverified Storage State</option>
                    </select>
                  </label>
                </>
              )}

              <label className="text-xs text-violet-200">
                API Key Environment Variable
                <input
                  className={`${inputStyle} mt-2`}
                  value={apiKeyEnv}
                  onChange={(e) => setApiKeyEnv(e.target.value)}
                  disabled={target === "guided" || target === "ollama"}
                />
                <small className="mt-1 block text-violet-300/60">Key read from backend environment; secret is never stored.</small>
              </label>

              <label className="text-xs text-violet-200">
                Group B: Never-Exposed Reference Control
                <select
                  className={`${inputStyle} mt-2`}
                  value={referenceType}
                  onChange={(e) => setReferenceType(e.target.value)}
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} (Matched Reference)
                    </option>
                  ))}
                </select>
                <small className="mt-1 block text-violet-300/60">Tests if pretraining prior knowledge explains target retrieval.</small>
              </label>

              <label className="text-xs text-violet-200">
                Group C: Exposed-Without-Forgetting Control
                <select
                  className={`${inputStyle} mt-2`}
                  value={counterfactualType}
                  onChange={(e) => setCounterfactualType(e.target.value)}
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} (Counterfactual Target)
                    </option>
                  ))}
                </select>
                <small className="mt-1 block text-violet-300/60">Estimates counterfactual retention for causal unlearning attribution.</small>
              </label>
            </div>

            <button
              disabled={busy || !name.trim() || (target !== "guided" && !model.trim())}
              onClick={() => void create()}
              className="mt-6 rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-950/40 transition hover:bg-violet-500 disabled:opacity-40"
            >
              {busy ? "Initializing Audit…" : "Initialize Audit Workspace ↗"}
            </button>
          </section>
        ) : (
          <section className="rounded-2xl border border-violet-300/20 bg-violet-950/20 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="text-xs font-semibold text-amber-300">
                  ACTIVE AUDIT WORKSPACE · {audit.track_metadata?.track === "track_b_memory_deletion" ? "TRACK B (STORAGE DELETION)" : "TRACK A (MODEL UNLEARNING)"}
                </span>
                <h2 className="text-xl font-bold">{audit.name}</h2>
                <p className="mt-1 text-xs text-violet-200/70">
                  ID: <code className="text-violet-300">{audit.id}</code> · Model:{" "}
                  <b>{audit.target?.model || audit.target?.type}</b> · Scope:{" "}
                  <b>{audit.target?.operation_category || "blackbox_disclosure_test"}</b>
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setAudit(null);
                    setReport(null);
                  }}
                  className="rounded-lg border border-violet-300/30 px-3 py-1.5 text-xs text-violet-200 hover:bg-white/5"
                >
                  Switch / New Audit
                </button>
              </div>
            </div>
          </section>
        )}

        {audit && (
          <>
            {/* STEP 2: FACT & RETAINED UTILITY CONFIGURATION */}
            <section className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-2xl border border-violet-300/15 bg-white/[.03] p-5">
                <span className="text-xs font-bold uppercase tracking-wider text-amber-300">Step 2A</span>
                <h2 className="text-lg font-semibold">Define Target Fact to Forget</h2>
                <p className="mt-1 text-xs text-violet-100/70">
                  Target information to test. Must be synthetic canaries or authorized data.
                </p>

                <div className="mt-4 space-y-3">
                  <label className="block text-xs text-violet-200">
                    Target Fact String
                    <input
                      className={`${inputStyle} mt-1`}
                      value={fact}
                      onChange={(e) => setFact(e.target.value)}
                      placeholder="e.g. Who wrote Romeo and Juliet?"
                    />
                  </label>

                  <label className="block text-xs text-violet-200">
                    Expected Target Token / Answer
                    <input
                      className={`${inputStyle} mt-1`}
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder="e.g. William Shakespeare"
                    />
                  </label>

                  <label className="block text-xs text-violet-200">
                    Direct Probe Question
                    <input
                      className={`${inputStyle} mt-1`}
                      value={targetQuestion}
                      onChange={(e) => setTargetQuestion(e.target.value)}
                      placeholder="e.g. Who wrote Romeo and Juliet?"
                    />
                  </label>
                </div>
              </div>

              <div className="rounded-2xl border border-violet-300/15 bg-white/[.03] p-5">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-300">Step 2B</span>
                <h2 className="text-lg font-semibold">Define Retained-Utility Control</h2>
                <p className="mt-1 text-xs text-violet-100/70">
                  Unrelated information that SHOULD remain accessible to verify utility is preserved.
                </p>

                <div className="mt-4 space-y-3">
                  <label className="block text-xs text-violet-200">
                    Retained-Utility Question
                    <input
                      className={`${inputStyle} mt-1`}
                      value={retainedQ}
                      onChange={(e) => setRetainedQ(e.target.value)}
                      placeholder="e.g. What is the capital city of France?"
                    />
                  </label>

                  <label className="block text-xs text-violet-200">
                    Expected Retained Answer
                    <input
                      className={`${inputStyle} mt-1`}
                      value={retainedA}
                      onChange={(e) => setRetainedA(e.target.value)}
                      placeholder="e.g. Paris"
                    />
                  </label>

                  <button
                    disabled={busy || !fact.trim() || !answer.trim()}
                    onClick={() => void addFact()}
                    className="mt-2 w-full rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-500 disabled:opacity-40"
                  >
                    Add Fact Pair to Audit
                  </button>
                </div>

                <div className="mt-4 border-t border-violet-300/10 pt-3">
                  <span className="text-xs text-violet-300/70">Or bulk import via dataset:</span>
                  <input
                    className="mt-2 block w-full text-xs text-violet-100 file:mr-2 file:rounded-lg file:border-0 file:bg-violet-800 file:px-2.5 file:py-1.5 file:text-white"
                    type="file"
                    accept=".csv,.json,.jsonl"
                    onChange={(e) => void upload(e.target.files?.[0])}
                  />

                  {facts.length > 1 && !audit.phases?.baseline && (
                    <div className="mt-3 rounded-lg border border-violet-300/10 bg-black/30 p-2.5">
                      <div className="text-[11px] font-semibold text-violet-200">Dataset Studio: Slice / Filter Facts</div>
                      <div className="mt-2 flex items-center gap-2">
                        <label className="text-[10px] text-violet-300">
                          Start
                          <input
                            type="number"
                            className="mt-0.5 w-16 rounded border border-violet-300/20 bg-black px-1.5 py-1 text-xs"
                            value={filterStart}
                            onChange={(e) => setFilterStart(Number(e.target.value))}
                          />
                        </label>
                        <label className="text-[10px] text-violet-300">
                          Limit
                          <input
                            type="number"
                            className="mt-0.5 w-16 rounded border border-violet-300/20 bg-black px-1.5 py-1 text-xs"
                            value={filterLimit}
                            onChange={(e) => setFilterLimit(Number(e.target.value))}
                          />
                        </label>
                        <button
                          disabled={busy}
                          onClick={() => void filterDataset()}
                          className="mt-3.5 rounded border border-violet-300/30 bg-violet-600/30 px-2 py-1 text-[11px] font-medium text-violet-100 hover:bg-violet-600/50"
                        >
                          Apply Filter
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* STEP 3: PROBING STRATEGIES */}
            <section className="rounded-2xl border border-violet-300/15 bg-white/[.03] p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-300">Step 3 of 6</span>
                  <h2 className="text-lg font-semibold">Generate Multi-Angle Probes</h2>
                  <p className="mt-1 text-xs text-violet-100/70">
                    Diverse probes test whether the secret leaks across direct, paraphrased, indirect, and multi-turn queries.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    disabled={busy || facts.length === 0}
                    onClick={() => void generate()}
                    className="rounded-xl bg-violet-600 px-4 py-2 text-xs font-semibold hover:bg-violet-500 disabled:opacity-40"
                  >
                    Auto-Generate Probes
                  </button>
                  <button
                    disabled={busy || !questions.length}
                    onClick={() => void loadReport()}
                    className="rounded-xl border border-violet-300/30 px-3 py-2 text-xs text-violet-200 hover:bg-white/5"
                  >
                    ↻ Refresh Report
                  </button>
                </div>
              </div>

              {questions.length > 0 && (
                <div className="mt-4 grid gap-2">
                  {questions.map((q: any) => (
                    <details key={q.id} className="rounded-xl border border-violet-300/10 bg-black/30 p-3 text-xs">
                      <summary className="cursor-pointer font-medium text-violet-100">
                        <span className="mr-2 rounded bg-violet-800 px-1.5 py-0.5 text-[10px] uppercase text-violet-200">
                          {q.kind}
                        </span>
                        {q.question}
                      </summary>
                      <div className="mt-3 flex gap-2">
                        <input
                          className={inputStyle}
                          value={editValues[q.id] ?? q.question}
                          onChange={(e) => setEditValues((v) => ({ ...v, [q.id]: e.target.value }))}
                          disabled={!!audit.phases?.baseline}
                        />
                        <button
                          disabled={busy || !!audit.phases?.baseline || !(editValues[q.id] ?? q.question).trim()}
                          onClick={() => void editQuestion(q)}
                          className="rounded-lg border border-violet-300/30 px-3 py-1.5 text-xs text-violet-200 disabled:opacity-40"
                        >
                          Save
                        </button>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </section>

            {/* STEP 4 & 5: EXPERIMENTAL EXECUTION */}
            <section className="rounded-2xl border border-violet-300/15 bg-white/[.03] p-5">
              <span className="text-xs font-bold uppercase tracking-wider text-amber-300">Steps 4 & 5 of 6</span>
              <h2 className="text-lg font-semibold">Run Controlled Experiment & Document Operation</h2>
              <p className="mt-1 text-xs text-violet-100/70">
                1) Baseline Gate: Target must be verifiable before unlearning. 2) Record the forgetting operation. 3) Run Group A post-forget, Group B never-exposed reference, and Group C counterfactual controls.
              </p>

              <div className="mt-4">
                <label className="block text-xs text-violet-200">
                  Document Forgetting Operation & Notes
                  <textarea
                    className={`${inputStyle} mt-1.5 min-h-16`}
                    value={forgetNotes}
                    onChange={(e) => setForgetNotes(e.target.value)}
                  />
                </label>
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                <button
                  disabled={busy || !facts.length}
                  onClick={() => void expose()}
                  className="rounded-xl border border-amber-300/30 bg-amber-400/5 px-4 py-2.5 text-xs font-semibold text-amber-200 hover:bg-amber-400/10 disabled:opacity-40"
                >
                  Prepare Exposure Context
                </button>

                <button
                  disabled={busy || !facts.length || !questions.length}
                  onClick={() => void run("baseline")}
                  className="rounded-xl bg-violet-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-violet-500 disabled:opacity-40"
                >
                  1. Run Baseline Probes
                </button>

                <button
                  disabled={busy || !audit.phases?.baseline}
                  onClick={() => void mark()}
                  className="rounded-xl border border-amber-300/40 bg-amber-400/10 px-4 py-2.5 text-xs font-semibold text-amber-200 hover:bg-amber-400/20 disabled:opacity-40"
                >
                  2. Record Operation & Deletion
                </button>

                <button
                  disabled={busy || !audit.phases?.forget_record}
                  onClick={() => void run("post_forget")}
                  className="rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 px-4 py-2.5 text-xs font-semibold text-white hover:brightness-110 disabled:opacity-40"
                >
                  3. Run Group A (Post-Forget)
                </button>

                <button
                  disabled={busy || !questions.length}
                  onClick={() => void run("never_exposed")}
                  className="rounded-xl border border-emerald-300/30 bg-emerald-400/10 px-4 py-2.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-400/20 disabled:opacity-40"
                >
                  4. Run Group B (Never-Exposed Reference)
                </button>

                <button
                  disabled={busy || !questions.length}
                  onClick={() => void run("exposed_no_forget")}
                  className="rounded-xl border border-cyan-300/30 bg-cyan-400/10 px-4 py-2.5 text-xs font-semibold text-cyan-200 hover:bg-cyan-400/20 disabled:opacity-40"
                >
                  5. Run Group C (Counterfactual Control)
                </button>
              </div>

              {/* Guided Manual Response Input */}
              {target === "guided" && (
                <div className="mt-5 rounded-xl border border-violet-300/20 bg-black/40 p-4">
                  <span className="text-xs font-semibold text-violet-200">Guided Mode: Manual Response Entry</span>
                  <div className="mt-3 grid gap-3 sm:grid-cols-3">
                    <label className="text-xs text-violet-200">
                      Phase
                      <select className={`${inputStyle} mt-1`} value={phase} onChange={(e) => setPhase(e.target.value)}>
                        <option value="baseline">Baseline</option>
                        <option value="post_forget">Group A (Post-forget)</option>
                        <option value="never_exposed">Group B (Never-exposed control)</option>
                        <option value="exposed_no_forget">Group C (Counterfactual control)</option>
                      </select>
                    </label>

                    <label className="text-xs text-violet-200">
                      Question
                      <select className={`${inputStyle} mt-1`} value={manualQ} onChange={(e) => setManualQ(e.target.value)}>
                        <option value="">Select question</option>
                        {questions.map((q: any) => (
                          <option key={q.id} value={q.id}>
                            [{q.kind}] {q.question.slice(0, 70)}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="text-xs text-violet-200">
                      Paste Model Response
                      <textarea
                        className={`${inputStyle} mt-1 min-h-16`}
                        value={manualAnswer}
                        onChange={(e) => setManualAnswer(e.target.value)}
                      />
                    </label>
                  </div>
                  <button
                    disabled={busy || !manualQ || !manualAnswer.trim()}
                    onClick={() => void manualSubmit()}
                    className="mt-3 rounded-xl bg-violet-700 px-4 py-2 text-xs font-semibold text-white hover:bg-violet-600 disabled:opacity-40"
                  >
                    Save Response & Score
                  </button>
                </div>
              )}
            </section>

            {/* STEP 6: FORMAL AUDIT REPORT & PUBLICATION ARTIFACTS */}
            {report && (
              <section className="rounded-2xl border border-violet-400/30 bg-gradient-to-b from-violet-950/40 via-purple-950/20 to-black/80 p-6 shadow-2xl">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b border-violet-300/10 pb-4">
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider text-amber-300">
                      Step 6 of 6 · {report.audit_track === "track_b_memory_deletion" ? "Track B (Storage Deletion)" : "Track A (Model Unlearning)"} Evidence Report
                    </span>
                    <h2 className="mt-1 text-2xl font-bold">{report.name || "Audit Results"}</h2>
                    <p className="mt-0.5 text-xs text-violet-200/70">
                      ID: {report.audit_id} · Model: {report.target?.model || report.target?.type} · Track: <code className="text-violet-300">{report.audit_track}</code>
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => downloadExport("csv")}
                      className="rounded-lg border border-violet-300/30 bg-violet-600/20 px-3 py-2 text-xs font-semibold text-violet-100 hover:bg-violet-600/40"
                    >
                      ⇩ Export CSV
                    </button>
                    <button
                      onClick={() => downloadExport("json")}
                      className="rounded-lg border border-violet-300/30 bg-violet-600/20 px-3 py-2 text-xs font-semibold text-violet-100 hover:bg-violet-600/40"
                    >
                      ⇩ Export JSON (SHA-256)
                    </button>
                    <button
                      onClick={() => downloadExport("latex")}
                      className="rounded-lg border border-violet-300/30 bg-violet-600/20 px-3 py-2 text-xs font-semibold text-violet-100 hover:bg-violet-600/40"
                    >
                      ⇩ Download .tex
                    </button>
                    <button
                      onClick={copyLatexTable}
                      className="rounded-lg border border-amber-300/40 bg-amber-400/10 px-3 py-2 text-xs font-semibold text-amber-200 hover:bg-amber-400/20"
                    >
                      {copiedLatex ? "✓ Copied LaTeX Table" : "▤ Copy LaTeX Table"}
                    </button>
                  </div>
                </div>

                {/* SCIENTIFIC VALIDITY GATES WIDGET */}
                {report.validity_gates && (
                  <div
                    className={`mt-5 rounded-xl border p-4 ${
                      report.validity_gates.status === "PASS"
                        ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-200"
                        : report.validity_gates.status === "WARNING"
                        ? "border-amber-400/30 bg-amber-500/10 text-amber-200"
                        : "border-rose-400/40 bg-rose-500/15 text-rose-200"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-widest">
                        Scientific Validity Gate Status: <b>{report.validity_gates.status}</b>
                      </span>
                      <span className="text-xs font-medium opacity-80">{report.validity_gates.summary}</span>
                    </div>
                    <details className="mt-2.5">
                      <summary className="cursor-pointer text-xs font-semibold opacity-90">
                        View Gate-by-Gate Verification Breakdown (5 Gates)
                      </summary>
                      <div className="mt-2 space-y-1.5 text-xs opacity-90">
                        {report.validity_gates.checks?.map((c: any) => (
                          <div key={c.gate} className="flex items-start gap-2">
                            <span
                              className={`rounded px-1.5 py-0.2 text-[9px] font-bold ${
                                c.status === "PASS"
                                  ? "bg-emerald-400/20 text-emerald-200"
                                  : c.status === "WARNING"
                                  ? "bg-amber-400/20 text-amber-200"
                                  : "bg-rose-400/20 text-rose-200"
                              }`}
                            >
                              {c.status}
                            </span>
                            <span className="font-semibold capitalize">{c.gate.replace(/_/g, " ")}:</span>
                            <span>{c.detail}</span>
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}

                {/* Verdict Badge */}
                <div
                  className={`mt-4 rounded-xl border p-4 ${
                    report.verdict?.includes("RESIDUAL RETRIEVAL SIGNAL DETECTED")
                      ? "border-amber-400/40 bg-amber-400/10 text-amber-200"
                      : report.verdict?.includes("NO RESIDUAL DISCLOSURE")
                      ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-200"
                      : "border-violet-400/30 bg-violet-900/20 text-violet-200"
                  }`}
                >
                  <span className="text-[10px] font-bold uppercase tracking-widest">Formal Verdict</span>
                  <div className="mt-1 text-lg font-bold">{report.verdict}</div>
                  <p className="mt-1 text-xs opacity-80">{report.verdict_scope}</p>
                </div>

                {/* TRACK-SPECIFIC METRICS */}
                {report.audit_track === "track_b_memory_deletion" && report.track_b_metrics && (
                  <div className="mt-5 rounded-xl border border-amber-300/20 bg-amber-400/5 p-4">
                    <span className="text-xs font-semibold text-amber-200">Track B: Authoritative Storage Deletion & Boundary Verification</span>
                    <div className="mt-3 grid gap-3 sm:grid-cols-3 text-xs">
                      <div className="rounded-lg border border-amber-300/15 bg-black/40 p-3">
                        <small className="text-amber-300/70">Authoritative Store Status</small>
                        <b className="mt-1 block text-base text-amber-200 capitalize">
                          {report.track_b_metrics.authoritative_store_status?.replace(/_/g, " ")}
                        </b>
                        <span className="text-[10px] text-amber-300/60">Component: {report.track_b_metrics.storage_component}</span>
                      </div>
                      <div className="rounded-lg border border-amber-300/15 bg-black/40 p-3">
                        <small className="text-amber-300/70">Boundary Integrity</small>
                        <b className="mt-1 block text-base text-emerald-300">
                          {report.track_b_metrics.boundary_integrity?.intact ? "Intact" : "Degraded"}
                        </b>
                        <span className="text-[10px] text-violet-300/60">{report.track_b_metrics.boundary_integrity?.status}</span>
                      </div>
                      <div className="rounded-lg border border-amber-300/15 bg-black/40 p-3">
                        <small className="text-amber-300/70">Cache & Retrieval Consistency</small>
                        <p className="mt-1 text-[11px] text-violet-200">{report.track_b_metrics.cache_consistency_assessment}</p>
                      </div>
                    </div>
                  </div>
                )}

                {report.audit_track === "track_a_model_unlearning" && report.track_a_metrics && (
                  <div className="mt-5 rounded-xl border border-violet-300/20 bg-violet-900/10 p-4">
                    <span className="text-xs font-semibold text-violet-200">Track A: Model Checkpoint & Trade-Off Metrics</span>
                    <div className="mt-3 grid gap-3 sm:grid-cols-2 text-xs">
                      <div className="rounded-lg border border-violet-300/15 bg-black/40 p-3">
                        <small className="text-violet-300/70">Forget-Retain Trade-Off Index (FRTI)</small>
                        <b className="mt-1 block text-xl text-emerald-300">
                          {report.track_a_metrics.forget_retain_tradeoff_index?.score != null
                            ? (report.track_a_metrics.forget_retain_tradeoff_index.score * 100).toFixed(1) + "%"
                            : "—"}
                        </b>
                        <span className="text-[10px] text-violet-300/60">
                          Formula: UPR × (1 − RDR) · Normalized [0, 1]
                        </span>
                      </div>
                      <div className="rounded-lg border border-violet-300/15 bg-black/40 p-3">
                        <small className="text-violet-300/70">Model Checkpoints & Unlearning Algorithm</small>
                        <div className="mt-1 text-[11px] text-violet-200">
                          <b>Method:</b> {report.track_a_metrics.unlearning_method || "Unspecified"}<br />
                          <b>Pre-Unlearn:</b> <code>{report.track_a_metrics.checkpoints?.pre_unlearn || "base"}</code><br />
                          <b>Post-Unlearn:</b> <code>{report.track_a_metrics.checkpoints?.post_unlearn || "target"}</code>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3-GROUP ARCHITECTURE COMPARISON & WILSON 95% CIs */}
                <div className="mt-5">
                  <span className="text-xs font-semibold uppercase tracking-wider text-violet-200">
                    3-Group Experimental Comparison & 95% Wilson Confidence Intervals
                  </span>
                  <div className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    {/* Group A */}
                    <div className="rounded-xl border border-violet-300/15 bg-black/40 p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-violet-300">Group A (Post-Forget)</span>
                        <span className="rounded bg-violet-800/60 px-1.5 py-0.5 text-[9px] uppercase text-violet-200">Target</span>
                      </div>
                      <div className="mt-2 text-2xl font-bold">
                        {report.three_group_comparison?.group_a_post_forget?.disclosures ?? 0}
                        <span className="text-sm font-normal text-violet-400">
                          {" "}
                          / {report.three_group_comparison?.group_a_post_forget?.scored ?? 0}
                        </span>
                      </div>
                      <div className="mt-1 text-xs font-medium text-amber-300">
                        {report.three_group_comparison?.group_a_post_forget?.wilson?.ci_formatted || "—"}
                      </div>
                      <small className="mt-1 block text-[11px] text-violet-300/60">Residual disclosure rate (RDR)</small>
                    </div>

                    {/* Group B */}
                    <div className="rounded-xl border border-violet-300/15 bg-black/40 p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-violet-300">Group B (Never-Exposed)</span>
                        <span className="rounded bg-emerald-900/60 px-1.5 py-0.5 text-[9px] uppercase text-emerald-200">Witness</span>
                      </div>
                      <div className="mt-2 text-2xl font-bold">
                        {report.three_group_comparison?.group_b_never_exposed?.disclosures ?? 0}
                        <span className="text-sm font-normal text-violet-400">
                          {" "}
                          / {report.three_group_comparison?.group_b_never_exposed?.scored ?? 0}
                        </span>
                      </div>
                      <div className="mt-1 text-xs font-medium text-violet-200">
                        {report.three_group_comparison?.group_b_never_exposed?.wilson?.ci_formatted || "—"}
                      </div>
                      <small className="mt-1 block text-[11px] text-violet-300/60">Pretraining prior disclosure (FDR)</small>
                    </div>

                    {/* Group C */}
                    <div className="rounded-xl border border-violet-300/15 bg-black/40 p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-violet-300">Group C (Counterfactual)</span>
                        <span className="rounded bg-cyan-900/60 px-1.5 py-0.5 text-[9px] uppercase text-cyan-200">Control</span>
                      </div>
                      <div className="mt-2 text-2xl font-bold text-cyan-200">
                        {report.three_group_comparison?.group_c_exposed_no_forget?.disclosures ?? 0}
                        <span className="text-sm font-normal text-violet-400">
                          {" "}
                          / {report.three_group_comparison?.group_c_exposed_no_forget?.scored ?? 0}
                        </span>
                      </div>
                      <div className="mt-1 text-xs font-medium text-cyan-300">
                        {report.three_group_comparison?.group_c_exposed_no_forget?.wilson?.ci_formatted || "—"}
                      </div>
                      <small className="mt-1 block text-[11px] text-violet-300/60">Unmitigated retention rate (URR)</small>
                    </div>

                    {/* Retained Utility */}
                    <div className="rounded-xl border border-violet-300/15 bg-black/40 p-4">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-violet-300">Retained Utility</span>
                        <span className="rounded bg-emerald-900/60 px-1.5 py-0.5 text-[9px] uppercase text-emerald-200">
                          {report.retained_utility?.status || "untested"}
                        </span>
                      </div>
                      <div className="mt-2 text-2xl font-bold text-emerald-300">
                        {report.retained_utility?.preserved ?? 0}
                        <span className="text-sm font-normal text-violet-400"> / {report.retained_utility?.tested ?? 0}</span>
                      </div>
                      <div className="mt-1 text-xs font-medium text-emerald-300">
                        {report.retained_utility?.wilson?.ci_formatted || "—"}
                      </div>
                      <small className="mt-1 block text-[11px] text-violet-300/60">Utility preservation score (RUS)</small>
                    </div>
                  </div>
                </div>

                {/* CAUSAL DELTA IMPACT ASSESSMENT */}
                {report.three_group_comparison?.causal_impact && (
                  <div className="mt-5 rounded-xl border border-violet-300/20 bg-gradient-to-r from-violet-900/30 to-purple-900/30 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-violet-200">
                        Causal Unlearning Attribution (Δ = Group C Recovery − Group A Recovery)
                      </span>
                      <span
                        className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                          report.three_group_comparison.causal_impact.status === "strong_causal_reduction"
                            ? "bg-emerald-400/20 text-emerald-200 border border-emerald-400/30"
                            : report.three_group_comparison.causal_impact.status === "moderate_causal_reduction"
                            ? "bg-amber-400/20 text-amber-200 border border-amber-400/30"
                            : "bg-violet-800/40 text-violet-200 border border-violet-400/20"
                        }`}
                      >
                        {report.three_group_comparison.causal_impact.status?.replace(/_/g, " ").toUpperCase()}
                      </span>
                    </div>
                    <div className="mt-2 text-xl font-bold text-white">
                      Δ = {report.three_group_comparison.causal_impact.ci_formatted}
                    </div>
                    <p className="mt-1 text-xs text-violet-200/80">
                      {report.three_group_comparison.causal_impact.interpretation}
                    </p>
                  </div>
                )}

                {/* MULTI-ANGLE PROBE TAXONOMY BREAKDOWN */}
                {report.probe_breakdown && (
                  <div className="mt-5 rounded-xl border border-violet-300/15 bg-black/30 p-4">
                    <span className="text-xs font-semibold text-violet-200">
                      Multi-Angle Probing Breakdown (Direct, Paraphrase, Indirect, Multi-Turn)
                    </span>
                    <div className="mt-3 overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-violet-300/10 text-violet-300">
                            <th className="pb-2 font-medium">Probe Taxonomy</th>
                            <th className="pb-2 font-medium">Tested</th>
                            <th className="pb-2 font-medium">Post-Forget Leakage</th>
                            <th className="pb-2 font-medium">95% Wilson CI</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-violet-300/5">
                          {Object.entries(report.probe_breakdown).map(([kind, p]: any) => (
                            <tr key={kind}>
                              <td className="py-2 capitalize font-semibold text-violet-200">{kind.replace("_", " ")}</td>
                              <td className="py-2 text-violet-300">{p.total}</td>
                              <td className="py-2 text-amber-300 font-bold">{p.disclosures} / {p.scored}</td>
                              <td className="py-2 text-violet-300/80">{p.wilson?.ci_formatted || "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Paired 4-Quadrant Control Matrix */}
                {report.paired_comparison?.available && (
                  <div className="mt-5 rounded-xl border border-violet-300/15 bg-black/30 p-4">
                    <span className="text-xs font-semibold text-violet-200">Paired Control Witness Matrix (4-Quadrant)</span>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-center sm:grid-cols-4">
                      <div className="rounded-lg border border-violet-300/10 bg-white/[.02] p-2.5">
                        <small className="block text-[10px] text-violet-300/70">Post-Forget Only (Leakage)</small>
                        <b className="text-lg text-amber-300">{report.paired_comparison.outcomes?.post_forget_only_flagged ?? 0}</b>
                      </div>
                      <div className="rounded-lg border border-violet-300/10 bg-white/[.02] p-2.5">
                        <small className="block text-[10px] text-violet-300/70">Neither Flagged (Clean)</small>
                        <b className="text-lg text-emerald-300">{report.paired_comparison.outcomes?.neither_flagged ?? 0}</b>
                      </div>
                      <div className="rounded-lg border border-violet-300/10 bg-white/[.02] p-2.5">
                        <small className="block text-[10px] text-violet-300/70">Both Flagged (Pretrained)</small>
                        <b className="text-lg text-violet-200">{report.paired_comparison.outcomes?.both_flagged ?? 0}</b>
                      </div>
                      <div className="rounded-lg border border-violet-300/10 bg-white/[.02] p-2.5">
                        <small className="block text-[10px] text-violet-300/70">Reference Only</small>
                        <b className="text-lg text-violet-200">{report.paired_comparison.outcomes?.reference_only_flagged ?? 0}</b>
                      </div>
                    </div>
                  </div>
                )}

                {/* Operation Scope Statement */}
                <div className="mt-5 rounded-xl border border-violet-300/10 bg-white/[.02] p-4 text-xs text-violet-200/70">
                  <b>Operational Boundary:</b> {report.operation_documentation?.operational_boundary}
                </div>

                {/* Full Traces Inspection */}
                <details className="mt-4">
                  <summary className="cursor-pointer text-xs font-semibold text-violet-300">
                    Inspect Raw Probe Responses, Refusal Classifications & Multi-Group Trials ({report.rows?.length || 0} questions)
                  </summary>
                  <div className="mt-3 max-h-96 space-y-2 overflow-auto rounded-xl bg-black/50 p-3 text-xs">
                    {report.rows?.map((row: any) => (
                      <div key={row.question_id} className="rounded-lg border border-violet-300/10 p-2.5">
                        <div className="flex items-center justify-between text-violet-200">
                          <b>
                            [{row.kind}] {row.question}
                          </b>
                          <div className="flex gap-1.5">
                            {row.post_forget?.score?.disclosure_classification && (
                              <span className="rounded bg-violet-800/50 px-1.5 py-0.5 text-[9px] uppercase text-violet-200 border border-violet-400/20">
                                {row.post_forget.score.disclosure_classification.replace(/_/g, " ")}
                              </span>
                            )}
                            <span
                              className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                                row.post_forget_lexical_flag ? "bg-amber-400/20 text-amber-200" : "bg-emerald-400/20 text-emerald-200"
                              }`}
                            >
                              {row.kind === "retained_utility"
                                ? row.post_forget_lexical_flag
                                  ? "Utility Preserved"
                                  : "Utility Degraded"
                                : row.post_forget_lexical_flag
                                ? "Disclosed"
                                : "Not Disclosed"}
                            </span>
                          </div>
                        </div>
                        <div className="mt-2 text-violet-300/80 space-y-1">
                          <p>
                            <b className="text-violet-200">Group A (Post-Forget):</b> {row.post_forget?.response || "No response recorded"}
                          </p>
                          {row.never_exposed && (
                            <p>
                              <b className="text-violet-200">Group B (Never-Exposed Reference):</b> {row.never_exposed?.response || "No response recorded"}
                            </p>
                          )}
                          {row.exposed_no_forget && (
                            <p>
                              <b className="text-violet-200">Group C (Counterfactual Control):</b> {row.exposed_no_forget?.response || "No response recorded"}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              </section>
            )}
          </>
        )}

        <footer className="border-t border-violet-300/10 pt-4 text-xs text-violet-300/50">
          RENA Research Platform · Observable disclosure evaluation under tested conditions · Wilson 95% CIs · Not proof of physical weight deletion.
        </footer>
      </div>
    </main>
  );
}
