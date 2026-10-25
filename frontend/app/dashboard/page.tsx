"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/navigation";
import { useRouter } from "next/navigation";
import {
  Audit,
  Algorithm,
  ForgettingAuditSummary,
  UnifiedAuditRow,
} from "@/lib/types";
import {
  listAudits,
  listForgettingAudits,
  createAudit,
  API,
} from "@/lib/api";
import ThemePicker from "@/components/ThemePicker";

const CLASSICAL_ALGORITHMS: {
  value: Algorithm;
  label: string;
  short: string;
  desc: string;
  icon: string;
  prereq: string;
}[] = [
  { value: "fine_tuning", label: "Fine-Tuning", short: "FT", desc: "Retain-only adaptation baseline", icon: "↗", prereq: "Standard classification architecture" },
  { value: "gradient_ascent", label: "Gradient Ascent", short: "GA", desc: "Pushes loss up on forget set", icon: "⌁", prereq: "Differentiable loss with learning rate control" },
  { value: "ground_truth_retraining", label: "Ground-Truth Retraining", short: "RETRAIN", desc: "Oracle model trained without forget data", icon: "◎", prereq: "Complete retain-only training compute budget" },
  { value: "sisa", label: "SISA", short: "SISA", desc: "Shard-based selective retraining", icon: "▦", prereq: "Shard-partitioned dataset index" },
  { value: "prism", label: "RENA-PR", short: "PRISM", desc: "Projected recovery-aware unlearning", icon: "✳", prereq: "Feature covariance tracking" },
  { value: "echo", label: "ECHO", short: "ECHO", desc: "Evidence-guided closed-loop updates", icon: "◉", prereq: "Witness feedback loop" },
  { value: "neggrad", label: "NegGrad+", short: "NG+", desc: "Selective negative-gradient forgetting", icon: "↘", prereq: "Forget/retain dual gradient computation" },
  { value: "salun", label: "SalUn", short: "SAL", desc: "Saliency-guided selective amnesia", icon: "✦", prereq: "Weight saliency gradient mask" },
  { value: "unsir", label: "UNSIR", short: "UNS", desc: "Synthetic-noise forgetting baseline", icon: "≈", prereq: "Input noise generator & retain repair" },
  { value: "scrub", label: "SCRUB", short: "SCR", desc: "Forget ascent + retain distillation", icon: "⊘", prereq: "Teacher model for retain distillation" },
];

const classicalLabel = (v: string) =>
  CLASSICAL_ALGORITHMS.find((a) => a.value === v)?.label ?? v.replaceAll("_", " ");

export default function DashboardPage() {
  const router = useRouter();

  // Audit data state
  const [forgettingAudits, setForgettingAudits] = useState<ForgettingAuditSummary[]>([]);
  const [classicalAudits, setClassicalAudits] = useState<Audit[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Modals & Drawers
  const [showNewAuditModal, setShowNewAuditModal] = useState(false);
  const [showAdvancedTools, setShowAdvancedTools] = useState(false);
  const [runningClassical, setRunningClassical] = useState<Algorithm | null>(null);
  const [selectedClassical, setSelectedClassical] = useState<Algorithm | null>(null);
  const [forgetFraction, setForgetFraction] = useState(10);
  const [splitSeed, setSplitSeed] = useState("42");

  // Filtering & Sorting
  const [searchQuery, setSearchQuery] = useState("");
  const [trackFilter, setTrackFilter] = useState<"all" | "track_a" | "track_b" | "classical">("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [verdictFilter, setVerdictFilter] = useState("all");
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest");
  const [showRecentActivity, setShowRecentActivity] = useState(false);

  // Refresh data
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Check health
      try {
        const hRes = await fetch(`${API}/health`, { cache: "no-store" });
        setApiOnline(hRes.ok);
      } catch {
        setApiOnline(false);
      }

      // Fetch both audit sets in parallel
      const [faResult, caResult] = await Promise.allSettled([
        listForgettingAudits(),
        listAudits(50),
      ]);

      if (faResult.status === "fulfilled") {
        setForgettingAudits(faResult.value);
      } else {
        console.warn("Forgetting audits fetch error:", faResult.reason);
      }

      if (caResult.status === "fulfilled") {
        setClassicalAudits(caResult.value);
      } else {
        console.warn("Classical audits fetch error:", caResult.reason);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Handle Classical ML Audit Launch
  async function handleRunClassical(algorithm: Algorithm) {
    setRunningClassical(algorithm);
    setError(null);
    try {
      await createAudit(algorithm, forgetFraction / 100, Number(splitSeed));
      await refresh();
      setShowAdvancedTools(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunningClassical(null);
    }
  }

  // Combine into unified audit rows
  const unifiedAudits: UnifiedAuditRow[] = useMemo(() => {
    const list: UnifiedAuditRow[] = [];

    // Forgetting Audits
    for (const fa of forgettingAudits) {
      list.push({
        id: fa.id,
        name: fa.name || "Controlled Forgetting Audit",
        audit_track: fa.audit_track,
        track_label:
          fa.audit_track === "track_b_memory_deletion"
            ? "Track B: Memory Deletion"
            : "Track A: Model Unlearning",
        status: fa.status,
        validity_status: fa.validity_status,
        primary_outcome: fa.primary_outcome,
        verdict: fa.verdict,
        created_at: fa.created_at,
        link: `/forgetting-audit?audit_id=${fa.id}`,
        raw_type: "forgetting",
      });
    }

    // Classical Audits
    for (const ca of classicalAudits) {
      list.push({
        id: ca.id,
        name: `${classicalLabel(ca.algorithm)} Classical Audit`,
        audit_track: "classical",
        track_label: "Classical ML",
        status: ca.status === "complete" ? "complete" : ca.status,
        validity_status: undefined,
        primary_outcome: ca.consensus_verdict
          ? `${ca.consensus_verdict} (Score: ${ca.consensus_score?.toFixed(3) ?? "—"})`
          : "Baseline Retrain Split",
        verdict: ca.consensus_verdict,
        created_at: ca.created_at,
        link: `/audits/${ca.id}`,
        raw_type: "classical",
      });
    }

    // Sort
    return list.sort((a, b) => {
      const ta = new Date(a.created_at).getTime();
      const tb = new Date(b.created_at).getTime();
      return sortOrder === "newest" ? tb - ta : ta - tb;
    });
  }, [forgettingAudits, classicalAudits, sortOrder]);

  // Filtered rows
  const filteredAudits = useMemo(() => {
    return unifiedAudits.filter((row) => {
      // Search
      const text = `${row.name} ${row.id} ${row.primary_outcome} ${row.track_label}`.toLowerCase();
      if (searchQuery && !text.includes(searchQuery.toLowerCase())) return false;

      // Track Filter
      if (trackFilter === "track_a" && row.audit_track !== "track_a_model_unlearning") return false;
      if (trackFilter === "track_b" && row.audit_track !== "track_b_memory_deletion") return false;
      if (trackFilter === "classical" && row.audit_track !== "classical") return false;

      // Status Filter
      if (statusFilter !== "all" && row.status !== statusFilter) return false;

      // Verdict Filter
      if (verdictFilter !== "all" && row.verdict !== verdictFilter) return false;

      return true;
    });
  }, [unifiedAudits, searchQuery, trackFilter, statusFilter, verdictFilter]);

  // Aggregate Metrics for the 4 Summary Cards
  const totalCount = unifiedAudits.length;
  const completedCount = unifiedAudits.filter((a) => a.status === "complete").length;
  const runningCount = unifiedAudits.filter(
    (a) => a.status === "running" || a.status === "pending" || a.status === "setup"
  ).length;
  const latestAudit = unifiedAudits[0] || null;

  // Verdict Analytics (Only computed and shown when completed audits exist)
  const completedAudits = useMemo(() => unifiedAudits.filter((a) => a.status === "complete"), [unifiedAudits]);
  const verdictDistribution = useMemo(() => {
    if (completedAudits.length === 0) return null;

    const noRecovery = completedAudits.filter(
      (a) =>
        a.verdict === "No target information recovered by tested probes" ||
        a.verdict === "Likely Forgotten"
    ).length;

    const targetRecovered = completedAudits.filter(
      (a) =>
        a.verdict === "Target information recovered" ||
        a.verdict === "Not Forgotten"
    ).length;

    const ambiguous = completedAudits.filter(
      (a) => a.verdict === "Ambiguous evidence" || a.verdict === "Ambiguous"
    ).length;

    const inconclusive = completedAudits.filter(
      (a) =>
        a.verdict === "Inconclusive or invalid experiment" ||
        a.verdict === "Inconclusive" ||
        a.verdict === "Inconclusive / Incomplete" ||
        (!a.verdict && a.status === "complete")
    ).length;

    return { noRecovery, targetRecovered, ambiguous, inconclusive, total: completedAudits.length };
  }, [completedAudits]);

  // Extract recent activity events from forgetting audits
  const recentEvents = useMemo(() => {
    const events: { at: string; event: string; auditName: string; auditId: string }[] = [];
    for (const fa of forgettingAudits) {
      if (fa.events && Array.isArray(fa.events)) {
        for (const ev of fa.events) {
          events.push({
            at: ev.at,
            event: ev.event.replaceAll("_", " "),
            auditName: fa.name,
            auditId: fa.id,
          });
        }
      }
    }
    return events
      .sort((a, b) => new Date(b.at).getTime() - new Date(a.at).getTime())
      .slice(0, 6);
  }, [forgettingAudits]);

  // Export CSV of filtered rows
  function exportCSV() {
    const headers = ["Audit ID", "Name", "Track", "Status", "Validity Gate", "Primary Outcome", "Verdict", "Created At"];
    const rows = filteredAudits.map((a) => [
      a.id,
      `"${a.name.replaceAll('"', '""')}"`,
      a.track_label,
      a.status,
      a.validity_status || "N/A",
      `"${a.primary_outcome.replaceAll('"', '""')}"`,
      `"${(a.verdict || "N/A").replaceAll('"', '""')}"`,
      a.created_at,
    ]);
    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `rena_audit_overview_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  // Export JSON of filtered rows
  function exportJSON() {
    const dataStr = JSON.stringify(filteredAudits, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `rena_audit_overview_${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="rena-shell min-h-screen text-white bg-[#0c0814]">
      {/* Sidebar Navigation */}
      <aside className="rena-sidebar">
        <a href="/" className="brand-lockup">
          <span className="brand-mark">R</span>
          <span>
            <b>RENA</b>
            <small>UNLEARNING AUDIT LAB</small>
          </span>
        </a>

        <div className="side-caption">WORKSPACE</div>
        <a href="/dashboard" className="side-link active">
          <span>◫</span> Overview
        </a>
        <a href="/forgetting-audit" className="side-link font-semibold text-amber-300">
          <span>✦</span> Audit Studio
        </a>
        <a href="/llm" className="side-link">
          <span>✧</span> LLM experiments
        </a>
        <a href="/mira" className="side-link">
          <span>◈</span> MIRA memory scoring
        </a>
        <a href="/attack-lab" className="side-link">
          <span>⌁</span> Adversarial lab
        </a>
        <a href="/security" className="side-link">
          <span>🛡</span> Security & integrity
        </a>

        <div className="side-bottom">
          <div className={`status-dot ${apiOnline === true ? "bg-emerald-400" : apiOnline === false ? "bg-red-400" : "bg-amber-400"}`} />
          <span>Research environment</span>
          <small>Audit & Evaluation Lab</small>
        </div>
      </aside>

      {/* Main Content Area */}
      <section className="rena-main">
        {/* Simplified Top Header */}
        <header className="topbar flex items-center justify-between border-b border-white/10 px-8 py-5">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Overview</h1>
            <p className="mt-0.5 text-xs text-zinc-400">
              Monitor your forgetting audits, review evidence, and start a new experiment.
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Live API Status indicator */}
            <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs">
              <span
                className={`h-2 w-2 rounded-full ${
                  apiOnline === true
                    ? "bg-emerald-400 animate-pulse"
                    : apiOnline === false
                    ? "bg-rose-500"
                    : "bg-amber-400"
                }`}
              />
              <span className="text-zinc-300 font-medium">
                {apiOnline === true ? "API Connected" : apiOnline === false ? "API Offline" : "Checking..."}
              </span>
            </div>

            <ThemePicker />

            <button
              onClick={() => void refresh()}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-white/10 transition"
              title="Refresh dashboard data"
            >
              <span className={loading ? "spin" : ""}>↻</span>
              <span>{loading ? "Refreshing" : "Refresh"}</span>
            </button>

            <span className="flex h-8 w-8 items-center justify-center rounded-full border border-violet-500/40 bg-violet-950/60 text-xs font-semibold text-violet-200">
              RS
            </span>
          </div>
        </header>

        <div className="page-wrap p-8 max-w-7xl mx-auto space-y-8">
          {/* Connection Error Banner */}
          {error && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-950/40 p-4 text-xs text-rose-200 flex items-center justify-between">
              <div>
                <b className="font-semibold">Connection issue:</b> {error}. Please ensure the backend container is running.
              </div>
              <button
                onClick={() => void refresh()}
                className="rounded-lg bg-rose-600/30 px-3 py-1 text-xs font-medium hover:bg-rose-600/50"
              >
                Retry ↻
              </button>
            </div>
          )}

          {/* Compact Hero Section */}
          <section className="rounded-2xl border border-violet-500/25 bg-gradient-to-br from-violet-950/40 via-[#130d22] to-black/80 p-7 shadow-2xl relative overflow-hidden">
            <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-violet-600/10 blur-3xl pointer-events-none" />
            <div className="flex flex-wrap items-center justify-between gap-6 relative z-10">
              <div className="max-w-2xl">
                <div className="inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-[11px] font-semibold text-amber-300">
                  <span>✦</span> RENA AUDIT FRAMEWORK
                </div>
                <h2 className="mt-2.5 text-2xl font-bold tracking-tight text-white">
                  RENA Audit Studio
                </h2>
                <p className="mt-1.5 text-sm leading-relaxed text-zinc-300">
                  Evaluate residual information recovery, retained utility, and deletion integrity using evidence-backed experiments.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setShowNewAuditModal(true)}
                  className="rounded-xl border border-amber-300/40 bg-gradient-to-r from-violet-600 to-amber-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-950/50 transition hover:brightness-110 active:scale-95"
                >
                  + New Audit
                </button>
                <a
                  href="#history"
                  className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-medium text-zinc-300 hover:bg-white/10 transition"
                >
                  View Audit History
                </a>
              </div>
            </div>
          </section>

          {/* Four Compact Summary Cards */}
          <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" aria-label="Overview summary">
            {/* 1. Total Audits */}
            <article className="rounded-xl border border-white/10 bg-[#120d1c] p-5 shadow-sm">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Total Audits</span>
                <span className="text-zinc-500 text-sm">▤</span>
              </div>
              <div className="mt-2 text-3xl font-bold tracking-tight text-white">
                {totalCount}
              </div>
              <p className="mt-1 text-[11px] text-zinc-500">
                {totalCount === 0 ? "No audits recorded" : "Across all evaluation tracks"}
              </p>
            </article>

            {/* 2. Completed Audits */}
            <article className="rounded-xl border border-white/10 bg-[#120d1c] p-5 shadow-sm">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Completed Audits</span>
                <span className="text-emerald-400 text-sm">✓</span>
              </div>
              <div className="mt-2 text-3xl font-bold tracking-tight text-emerald-400">
                {completedCount}
              </div>
              <p className="mt-1 text-[11px] text-zinc-500">
                {completedCount === 0 ? "Zero completed runs" : "Audits with final evidence ledger"}
              </p>
            </article>

            {/* 3. Running Audits */}
            <article className="rounded-xl border border-white/10 bg-[#120d1c] p-5 shadow-sm">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Running / Pending</span>
                <span className="text-amber-400 text-sm">◷</span>
              </div>
              <div className="mt-2 text-3xl font-bold tracking-tight text-amber-300">
                {runningCount}
              </div>
              <p className="mt-1 text-[11px] text-zinc-500">
                {runningCount === 0 ? "No active evaluations" : "In-flight operations or setups"}
              </p>
            </article>

            {/* 4. Latest Audit */}
            <article className="rounded-xl border border-white/10 bg-[#120d1c] p-5 shadow-sm">
              <div className="flex items-center justify-between text-xs text-zinc-400">
                <span>Latest Audit</span>
                <span className="text-violet-400 text-sm">◈</span>
              </div>
              {latestAudit ? (
                <>
                  <div className="mt-2 text-base font-semibold text-white truncate" title={latestAudit.name}>
                    {latestAudit.name}
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-zinc-400">
                    <span className="rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] text-violet-300">
                      {latestAudit.audit_track === "track_b_memory_deletion"
                        ? "Track B"
                        : latestAudit.audit_track === "track_a_model_unlearning"
                        ? "Track A"
                        : "Classical"}
                    </span>
                    <span>{new Date(latestAudit.created_at).toLocaleDateString()}</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="mt-2 text-base font-medium text-zinc-500">
                    No activity yet
                  </div>
                  <p className="mt-1 text-[11px] text-zinc-500">
                    Waiting for first audit
                  </p>
                </>
              )}
            </article>
          </section>

          {/* Small Verdict Summary (ONLY shown when completed audits exist) */}
          {verdictDistribution && verdictDistribution.total > 0 && (
            <section className="rounded-xl border border-white/10 bg-[#120d1c] p-5 shadow-sm space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                    Evidence-Based Verdict Distribution
                  </h3>
                  <p className="text-[11px] text-zinc-500">
                    Calculated strictly from completed audit evidence. Unsuccessful probes demonstrate behavioral non-recovery, not proof of complete parameter erasure.
                  </p>
                </div>
                <span className="text-xs text-zinc-400 font-medium">
                  {verdictDistribution.total} completed {verdictDistribution.total === 1 ? "audit" : "audits"}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-1">
                {/* 1. No target information recovered */}
                <button
                  onClick={() => {
                    setVerdictFilter("No target information recovered by tested probes");
                    document.getElementById("history")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className={`rounded-lg border p-3 text-left transition ${
                    verdictFilter === "No target information recovered by tested probes"
                      ? "border-emerald-500/60 bg-emerald-950/30"
                      : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-emerald-300">
                      <span className="h-2 w-2 rounded-full bg-emerald-400" />
                      No recovery observed
                    </span>
                    <b className="text-emerald-400">{verdictDistribution.noRecovery}</b>
                  </div>
                  <p className="mt-1 text-[10px] text-zinc-400 leading-tight">
                    Tested probes yielded zero target disclosure.
                  </p>
                </button>

                {/* 2. Target information recovered */}
                <button
                  onClick={() => {
                    setVerdictFilter("Target information recovered");
                    document.getElementById("history")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className={`rounded-lg border p-3 text-left transition ${
                    verdictFilter === "Target information recovered"
                      ? "border-rose-500/60 bg-rose-950/30"
                      : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-rose-300">
                      <span className="h-2 w-2 rounded-full bg-rose-400" />
                      Target recovered
                    </span>
                    <b className="text-rose-400">{verdictDistribution.targetRecovered}</b>
                  </div>
                  <p className="mt-1 text-[10px] text-zinc-400 leading-tight">
                    Residual target leaked in one or more probes.
                  </p>
                </button>

                {/* 3. Ambiguous evidence */}
                <button
                  onClick={() => {
                    setVerdictFilter("Ambiguous evidence");
                    document.getElementById("history")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className={`rounded-lg border p-3 text-left transition ${
                    verdictFilter === "Ambiguous evidence"
                      ? "border-amber-500/60 bg-amber-950/30"
                      : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-amber-300">
                      <span className="h-2 w-2 rounded-full bg-amber-400" />
                      Ambiguous
                    </span>
                    <b className="text-amber-400">{verdictDistribution.ambiguous}</b>
                  </div>
                  <p className="mt-1 text-[10px] text-zinc-400 leading-tight">
                    Partial refusal or conflicting verifier signals.
                  </p>
                </button>

                {/* 4. Inconclusive or invalid */}
                <button
                  onClick={() => {
                    setVerdictFilter("Inconclusive or invalid experiment");
                    document.getElementById("history")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className={`rounded-lg border p-3 text-left transition ${
                    verdictFilter === "Inconclusive or invalid experiment"
                      ? "border-zinc-500/60 bg-zinc-900/40"
                      : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 font-medium text-zinc-300">
                      <span className="h-2 w-2 rounded-full bg-zinc-400" />
                      Inconclusive / Invalid
                    </span>
                    <b className="text-zinc-300">{verdictDistribution.inconclusive}</b>
                  </div>
                  <p className="mt-1 text-[10px] text-zinc-400 leading-tight">
                    Baseline failed or validity gates unmet.
                  </p>
                </button>
              </div>

              {verdictFilter !== "all" && (
                <div className="flex items-center justify-between pt-1">
                  <span className="text-[11px] text-zinc-400">
                    Filtered by verdict: <b className="text-white">{verdictFilter}</b>
                  </span>
                  <button
                    onClick={() => setVerdictFilter("all")}
                    className="text-[11px] text-amber-300 hover:underline"
                  >
                    Clear verdict filter ×
                  </button>
                </div>
              )}
            </section>
          )}

          {/* Audit History Section (Main Content Area) */}
          <section id="history" className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white tracking-tight">Audit History</h2>
                <p className="text-xs text-zinc-400">
                  Traceable experiment records across model unlearning, memory deletion, and classical baselines.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => void refresh()}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-zinc-300 hover:bg-white/10 transition"
                  title="Reload audit records"
                >
                  ↻ Refresh
                </button>
                <button
                  onClick={exportCSV}
                  disabled={filteredAudits.length === 0}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-zinc-300 hover:bg-white/10 transition disabled:opacity-40"
                >
                  ⇩ Export CSV
                </button>
                <button
                  onClick={exportJSON}
                  disabled={filteredAudits.length === 0}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-zinc-300 hover:bg-white/10 transition disabled:opacity-40"
                >
                  ⇩ Export JSON
                </button>
              </div>
            </div>

            {/* Filter & Search Bar */}
            <div className="rounded-xl border border-white/10 bg-[#120d1c] p-3.5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                {/* Search Box */}
                <div className="flex-1 min-w-[260px] relative">
                  <span className="absolute left-3 top-2.5 text-xs text-zinc-500">⌕</span>
                  <input
                    type="text"
                    aria-label="Search audits"
                    placeholder="Search by audit name, ID, or outcome..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full rounded-lg border border-white/10 bg-black/40 pl-8 pr-8 py-2 text-xs text-white placeholder-zinc-500 focus:border-violet-500 focus:outline-none"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => setSearchQuery("")}
                      className="absolute right-3 top-2 text-xs text-zinc-400 hover:text-white"
                    >
                      ×
                    </button>
                  )}
                </div>

                {/* Track Filter Pills */}
                <div className="flex items-center rounded-lg border border-white/10 bg-black/40 p-1 text-xs">
                  <button
                    onClick={() => setTrackFilter("all")}
                    className={`rounded px-2.5 py-1 transition ${
                      trackFilter === "all" ? "bg-violet-600 text-white font-medium" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    All Tracks
                  </button>
                  <button
                    onClick={() => setTrackFilter("track_a")}
                    className={`rounded px-2.5 py-1 transition ${
                      trackFilter === "track_a" ? "bg-violet-600 text-white font-medium" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    Track A: Model
                  </button>
                  <button
                    onClick={() => setTrackFilter("track_b")}
                    className={`rounded px-2.5 py-1 transition ${
                      trackFilter === "track_b" ? "bg-violet-600 text-white font-medium" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    Track B: Memory
                  </button>
                  <button
                    onClick={() => setTrackFilter("classical")}
                    className={`rounded px-2.5 py-1 transition ${
                      trackFilter === "classical" ? "bg-violet-600 text-white font-medium" : "text-zinc-400 hover:text-white"
                    }`}
                  >
                    Classical ML
                  </button>
                </div>

                {/* Status Dropdown */}
                <select
                  aria-label="Filter by status"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-xs text-zinc-300 focus:border-violet-500 focus:outline-none"
                >
                  <option value="all">All Statuses</option>
                  <option value="complete">Completed</option>
                  <option value="running">Running</option>
                  <option value="pending">Pending</option>
                  <option value="setup">Setup</option>
                  <option value="failed">Failed</option>
                </select>

                {/* Sort Order */}
                <button
                  onClick={() => setSortOrder(sortOrder === "newest" ? "oldest" : "newest")}
                  className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5 transition flex items-center gap-1.5"
                  title="Toggle creation date order"
                >
                  <span>{sortOrder === "newest" ? "↓ Newest" : "↑ Oldest"}</span>
                </button>
              </div>
            </div>

            {/* Audit Records Table */}
            <div className="rounded-xl border border-white/10 bg-[#120d1c] overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-white/10 bg-white/[0.02] text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                    <tr>
                      <th className="py-3 px-4">Audit Name / ID</th>
                      <th className="py-3 px-4">Track</th>
                      <th className="py-3 px-4">Status</th>
                      <th className="py-3 px-4">Scientific Validity</th>
                      <th className="py-3 px-4">Primary Outcome</th>
                      <th className="py-3 px-4">Created</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-zinc-300">
                    {loading && unifiedAudits.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-zinc-500">
                          <span className="loader inline-block mr-2" /> Loading audit history...
                        </td>
                      </tr>
                    ) : filteredAudits.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-14 text-center">
                          <div className="max-w-md mx-auto space-y-3">
                            <div className="h-10 w-10 mx-auto rounded-full bg-violet-950/60 border border-violet-500/30 flex items-center justify-center text-violet-300 text-lg">
                              ⌁
                            </div>
                            <h3 className="text-sm font-semibold text-white">
                              {unifiedAudits.length === 0
                                ? "No audits recorded yet"
                                : "No matching audits found"}
                            </h3>
                            <p className="text-xs text-zinc-400 leading-relaxed">
                              {unifiedAudits.length === 0
                                ? "Create your first audit to collect traceable evidence of model unlearning or application memory deletion."
                                : "Try clearing your search query or switching your track and status filters."}
                            </p>
                            {unifiedAudits.length === 0 ? (
                              <button
                                onClick={() => setShowNewAuditModal(true)}
                                className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-amber-300/40 bg-gradient-to-r from-violet-600 to-amber-600 px-4 py-2 text-xs font-semibold text-white shadow hover:brightness-110"
                              >
                                + Start Your First Audit
                              </button>
                            ) : (
                              <button
                                onClick={() => {
                                  setSearchQuery("");
                                  setTrackFilter("all");
                                  setStatusFilter("all");
                                  setVerdictFilter("all");
                                }}
                                className="mt-2 text-xs text-amber-300 hover:underline"
                              >
                                Clear all filters
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ) : (
                      filteredAudits.map((a) => (
                        <tr key={a.id} className="hover:bg-white/[0.02] transition">
                          {/* Name & ID */}
                          <td className="py-3.5 px-4">
                            <a href={a.link} className="group block">
                              <div className="font-semibold text-white group-hover:text-amber-300 transition">
                                {a.name}
                              </div>
                              <div className="text-[10px] text-zinc-500 font-mono">
                                #{a.id.slice(0, 10)}
                              </div>
                            </a>
                          </td>

                          {/* Track */}
                          <td className="py-3.5 px-4 whitespace-nowrap">
                            <span
                              className={`inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-medium border ${
                                a.audit_track === "track_b_memory_deletion"
                                  ? "border-teal-500/30 bg-teal-950/40 text-teal-300"
                                  : a.audit_track === "track_a_model_unlearning"
                                  ? "border-violet-500/30 bg-violet-950/40 text-violet-300"
                                  : "border-zinc-500/30 bg-zinc-900/60 text-zinc-400"
                              }`}
                            >
                              {a.track_label}
                            </span>
                          </td>

                          {/* Status */}
                          <td className="py-3.5 px-4 whitespace-nowrap">
                            <span
                              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium border ${
                                a.status === "complete"
                                  ? "border-emerald-500/30 bg-emerald-950/40 text-emerald-300"
                                  : a.status === "running"
                                  ? "border-amber-500/30 bg-amber-950/40 text-amber-300 animate-pulse"
                                  : a.status === "failed"
                                  ? "border-rose-500/30 bg-rose-950/40 text-rose-300"
                                  : "border-zinc-500/30 bg-zinc-900/50 text-zinc-400"
                              }`}
                            >
                              <i
                                className={`h-1.5 w-1.5 rounded-full ${
                                  a.status === "complete"
                                    ? "bg-emerald-400"
                                    : a.status === "running"
                                    ? "bg-amber-400"
                                    : a.status === "failed"
                                    ? "bg-rose-400"
                                    : "bg-zinc-500"
                                }`}
                              />
                              {a.status.charAt(0).toUpperCase() + a.status.slice(1)}
                            </span>
                          </td>

                          {/* Scientific Validity */}
                          <td className="py-3.5 px-4 whitespace-nowrap">
                            {a.validity_status === "PASS" ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400">
                                <span>✓</span> Validated
                              </span>
                            ) : a.validity_status === "WARNING" ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-400">
                                <span>⚠</span> Warnings
                              </span>
                            ) : a.validity_status === "BLOCKED" ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-rose-400">
                                <span>✕</span> Blocked
                              </span>
                            ) : (
                              <span className="text-[11px] text-zinc-500">—</span>
                            )}
                          </td>

                          {/* Primary Outcome */}
                          <td className="py-3.5 px-4 max-w-xs">
                            <div className="text-xs text-zinc-200 truncate" title={a.primary_outcome}>
                              {a.primary_outcome}
                            </div>
                          </td>

                          {/* Created */}
                          <td className="py-3.5 px-4 whitespace-nowrap text-[11px] text-zinc-400">
                            <div>{new Date(a.created_at).toLocaleDateString()}</div>
                            <div className="text-[10px] text-zinc-600">
                              {new Date(a.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </div>
                          </td>

                          {/* Actions */}
                          <td className="py-3.5 px-4 text-right whitespace-nowrap">
                            <a
                              href={a.link}
                              className="inline-flex items-center gap-1 rounded-md border border-amber-300/30 bg-amber-400/10 px-2.5 py-1 text-xs font-medium text-amber-200 hover:bg-amber-400/20 transition"
                            >
                              <span>Open</span>
                              <span>↗</span>
                            </a>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Table Footer */}
              <div className="border-t border-white/5 bg-white/[0.01] px-4 py-2.5 flex items-center justify-between text-[11px] text-zinc-500">
                <span>
                  Showing <b>{filteredAudits.length}</b> of <b>{unifiedAudits.length}</b> records
                </span>
                <span>Traceable audit ledger with cryptographic integrity</span>
              </div>
            </div>
          </section>

          {/* Compact Recent Activity (Rendered if event records exist) */}
          {recentEvents.length > 0 && (
            <section className="rounded-xl border border-white/10 bg-[#120d1c] p-4 space-y-3">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setShowRecentActivity(!showRecentActivity)}
                  className="flex items-center gap-2 text-xs font-semibold text-zinc-300 hover:text-white transition"
                >
                  <span>{showRecentActivity ? "▼" : "▶"}</span>
                  <span>Recent Experiment Activity ({recentEvents.length} events logged)</span>
                </button>
                <span className="text-[10px] text-zinc-500">Backend event stream</span>
              </div>

              {showRecentActivity && (
                <div className="divide-y divide-white/5 pt-1 text-xs text-zinc-300">
                  {recentEvents.map((ev, i) => (
                    <div key={i} className="py-2 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />
                        <span className="font-medium text-white capitalize">{ev.event}</span>
                        <span className="text-[11px] text-zinc-500">on {ev.auditName}</span>
                      </div>
                      <span className="text-[10px] text-zinc-500 font-mono">
                        {new Date(ev.at).toLocaleTimeString()}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Advanced Research Tools & Classical ML Baselines */}
          <section className="rounded-xl border border-white/10 bg-[#120d1c] p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold text-white">Advanced Research & Specialized Tools</h3>
                <p className="text-xs text-zinc-400">
                  Access classical machine learning unlearning baselines, adversarial red-teaming, MIRA scoring, and security controls.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowAdvancedTools(!showAdvancedTools)}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-white/10 transition"
                >
                  {showAdvancedTools ? "Hide Classical ML Baselines" : "Show Classical ML Baselines (FT, GA, SISA...)"}
                </button>
              </div>
            </div>

            {/* Collapsible Classical ML Runner */}
            {showAdvancedTools && (
              <div className="pt-3 border-t border-white/10 space-y-4">
                <div className="text-xs text-zinc-400 leading-relaxed">
                  Classical unlearning algorithms operate on tabular or vision classification datasets (e.g. Scikit-learn Digits).
                  Select a method to train an initial model, apply the unlearning transformation, and evaluate against retain-only reference retrains.
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
                  {CLASSICAL_ALGORITHMS.map((algo) => (
                    <button
                      key={algo.value}
                      onClick={() => setSelectedClassical(algo.value)}
                      disabled={runningClassical !== null}
                      className={`p-3 rounded-lg border text-left transition flex flex-col justify-between ${
                        selectedClassical === algo.value
                          ? "border-amber-400/60 bg-amber-400/10 text-white"
                          : "border-white/5 bg-black/30 hover:bg-white/5 text-zinc-300"
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-bold text-amber-300">{algo.short}</span>
                          <span className="text-zinc-500">{algo.icon}</span>
                        </div>
                        <div className="mt-1 font-medium text-[11px] truncate">{algo.label}</div>
                        <div className="mt-0.5 text-[10px] text-zinc-500 line-clamp-2">{algo.desc}</div>
                      </div>
                      <div className="mt-2 text-[9px] text-zinc-400 font-mono">
                        {selectedClassical === algo.value ? "✓ Selected" : "Click to select"}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Classical Settings & Launch */}
                {selectedClassical && (
                  <div className="rounded-lg border border-amber-400/30 bg-amber-950/20 p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <div className="text-xs font-semibold text-amber-200">
                          Selected Method: {classicalLabel(selectedClassical)}
                        </div>
                        <div className="text-[11px] text-zinc-400">
                          Configure reproducible parameters for this classical classification run.
                        </div>
                      </div>

                      <div className="flex items-center gap-4">
                        <label className="flex items-center gap-2 text-xs text-zinc-300">
                          <span>Forget %:</span>
                          <input
                            type="number"
                            min="5"
                            max="40"
                            step="5"
                            value={forgetFraction}
                            onChange={(e) => setForgetFraction(Number(e.target.value))}
                            className="w-16 rounded border border-white/10 bg-black/60 px-2 py-1 text-xs text-white"
                          />
                        </label>

                        <label className="flex items-center gap-2 text-xs text-zinc-300">
                          <span>Split Seed:</span>
                          <input
                            type="number"
                            value={splitSeed}
                            onChange={(e) => setSplitSeed(e.target.value)}
                            className="w-20 rounded border border-white/10 bg-black/60 px-2 py-1 text-xs text-white"
                          />
                        </label>

                        <button
                          onClick={() => void handleRunClassical(selectedClassical)}
                          disabled={runningClassical !== null}
                          className="rounded-lg bg-amber-500 px-4 py-2 text-xs font-semibold text-black hover:bg-amber-400 transition disabled:opacity-40"
                        >
                          {runningClassical ? "Executing..." : `Launch ${classicalLabel(selectedClassical)} ↗`}
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Quick links to secondary research tools */}
            <div className="flex flex-wrap gap-2 pt-1 text-xs">
              <a
                href="/attack-lab"
                className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-1.5 text-zinc-400 hover:text-white hover:bg-white/[0.06] transition"
              >
                ⌁ Adversarial Lab
              </a>
              <a
                href="/mira"
                className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-1.5 text-zinc-400 hover:text-white hover:bg-white/[0.06] transition"
              >
                ◈ MIRA Memory Scoring
              </a>
              <a
                href="/llm"
                className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-1.5 text-zinc-400 hover:text-white hover:bg-white/[0.06] transition"
              >
                ✧ LLM Experiments
              </a>
              <a
                href="/security"
                className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-1.5 text-zinc-400 hover:text-white hover:bg-white/[0.06] transition"
              >
                🛡 Security & Certificates
              </a>
              <a
                href="/production"
                className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-1.5 text-zinc-400 hover:text-white hover:bg-white/[0.06] transition"
              >
                ⚙ Production Controls
              </a>
            </div>
          </section>

          {/* Simple Clean Footer */}
          <footer className="pt-4 border-t border-white/10 flex flex-wrap items-center justify-between text-xs text-zinc-500">
            <span>RENA · Reliable Evaluation for Neural Unlearning Audits</span>
            <span>Research Platform v2.3.0</span>
          </footer>
        </div>
      </section>

      {/* New Audit Modal (Unified Track Selection) */}
      {showNewAuditModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="w-full max-w-2xl rounded-2xl border border-violet-500/30 bg-[#120d1c] p-7 shadow-2xl space-y-6 animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-semibold text-amber-300 uppercase tracking-wider">
                  START EXPERIMENT
                </span>
                <h3 className="text-xl font-bold text-white mt-0.5">Select Audit Track</h3>
              </div>
              <button
                onClick={() => setShowNewAuditModal(false)}
                className="rounded-lg p-1 text-zinc-400 hover:text-white hover:bg-white/10 transition text-lg"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-zinc-300 leading-relaxed">
              RENA enforces strict scientific separation between neural weight modifications and external storage purges. Choose the track matching your system:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Option 1: Track A */}
              <a
                href="/forgetting-audit?track=track_a_model_unlearning"
                className="group rounded-xl border border-violet-500/30 bg-violet-950/20 p-5 hover:border-violet-400/60 hover:bg-violet-950/40 transition flex flex-col justify-between space-y-3"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="rounded-md border border-violet-400/30 bg-violet-500/20 px-2 py-0.5 text-[10px] font-semibold text-violet-300">
                      Track A
                    </span>
                    <span className="text-violet-400 text-lg">⌁</span>
                  </div>
                  <h4 className="mt-2 text-base font-bold text-white group-hover:text-amber-300 transition">
                    Model-Level Unlearning
                  </h4>
                  <p className="mt-1 text-xs text-zinc-400 leading-relaxed">
                    Evaluate an original model and a genuinely unlearned model checkpoint (e.g. Gradient Ascent, NPO, SCRUB, LoRA). Measures residual parametric disclosure against retained utility.
                  </p>
                </div>
                <div className="pt-2 text-xs font-semibold text-violet-300 group-hover:text-amber-300 flex items-center gap-1">
                  <span>Launch Track A Audit</span>
                  <span>→</span>
                </div>
              </a>

              {/* Option 2: Track B */}
              <a
                href="/forgetting-audit?track=track_b_memory_deletion"
                className="group rounded-xl border border-teal-500/30 bg-teal-950/20 p-5 hover:border-teal-400/60 hover:bg-teal-950/40 transition flex flex-col justify-between space-y-3"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="rounded-md border border-teal-400/30 bg-teal-500/20 px-2 py-0.5 text-[10px] font-semibold text-teal-300">
                      Track B
                    </span>
                    <span className="text-teal-400 text-lg">▦</span>
                  </div>
                  <h4 className="mt-2 text-base font-bold text-white group-hover:text-amber-300 transition">
                    Application Memory & Retrieval
                  </h4>
                  <p className="mt-1 text-xs text-zinc-400 leading-relaxed">
                    Evaluate deletion from a vector database, session context, or RAG index. Assesses index accessibility and secondary retrieval paths without claiming model weight modifications.
                  </p>
                </div>
                <div className="pt-2 text-xs font-semibold text-teal-300 group-hover:text-amber-300 flex items-center gap-1">
                  <span>Launch Track B Audit</span>
                  <span>→</span>
                </div>
              </a>
            </div>

            <div className="pt-2 border-t border-white/10 flex items-center justify-between text-xs text-zinc-400">
              <span>Looking for classical ML benchmarks?</span>
              <button
                onClick={() => {
                  setShowNewAuditModal(false);
                  setShowAdvancedTools(true);
                }}
                className="text-amber-300 hover:underline font-medium"
              >
                Open Classical ML Baselines →
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
