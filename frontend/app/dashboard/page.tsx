"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Audit, Algorithm } from "@/lib/types";
import { listAudits, createAudit } from "@/lib/api";
import { VerdictBadge } from "@/components/VerdictBadge";
import ThemePicker from "@/components/ThemePicker";

const ALGORITHMS: { value: Algorithm; label: string; short: string; desc: string; icon: string }[] = [
  { value: "fine_tuning", label: "Fine-Tuning", short: "FT", desc: "Retain-only adaptation baseline", icon: "↗" },
  { value: "gradient_ascent", label: "Gradient Ascent", short: "GA", desc: "Pushes loss up on forget samples", icon: "⌁" },
  { value: "ground_truth_retraining", label: "Ground-Truth Retraining", short: "RETRAIN", desc: "Reference trained without forget data", icon: "◎" },
  { value: "sisa", label: "SISA", short: "SISA", desc: "Shard-based selective retraining", icon: "▦" },
  { value: "prism", label: "RENA-PR", short: "RENA-PR", desc: "Projected, recovery-aware unlearning", icon: "✳" },
  { value: "echo", label: "ECHO", short: "ECHO", desc: "Evidence-guided closed-loop updates", icon: "◉" },
  { value: "neggrad", label: "NegGrad+", short: "NG+", desc: "Selective negative-gradient forgetting", icon: "↘" },
  { value: "salun", label: "SalUn", short: "SAL", desc: "Saliency-guided selective amnesia", icon: "✦" },
  { value: "unsir", label: "UNSIR", short: "UNS", desc: "Synthetic-noise forgetting baseline", icon: "≈" },
  { value: "scrub", label: "SCRUB", short: "SCR", desc: "Forget ascent + retain distillation", icon: "⊘" },
];
const label = (v: string) => ALGORITHMS.find(a => a.value === v)?.label ?? v.replaceAll("_", " ");
const pct = (v: number | null) => v == null ? "—" : `${(v * 100).toFixed(1)}%`;

export default function DashboardPage() {

  const [audits, setAudits] = useState<Audit[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<Algorithm | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<Algorithm | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [forgetFraction, setForgetFraction] = useState(10);
  const [splitSeed, setSplitSeed] = useState("42");
  const [chartMetric, setChartMetric] = useState<"runs" | "agreement">("runs");
  const [compareLeft, setCompareLeft] = useState("");
  const [compareRight, setCompareRight] = useState("");
  const refresh = useCallback(async () => {
    try { setAudits(await listAudits()); setError(null); }
    catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  async function handleRunAudit(algorithm: Algorithm) {
    setRunning(algorithm); setError(null);
    try { await createAudit(algorithm, forgetFraction / 100, Number(splitSeed)); await refresh(); }
    catch (e) { setError((e as Error).message); }
    finally { setRunning(null); }
  }
  const filtered = useMemo(() => audits.filter(a => {
    const matches = `${label(a.algorithm)} ${a.id} ${(a.authoritative_verdict ?? a.rena_x_verdict ?? a.consensus_verdict ?? "")}`.toLowerCase().includes(query.toLowerCase());
    return matches && (filter === "all" || a.status === filter);
  }), [audits, query, filter]);
  const complete = audits.filter(a => a.status === "complete");
  const avgAgreement = complete.length ? complete.reduce((n,a) => n + (a.consensus_agreement ?? 0), 0) / complete.length : null;
  const latest = audits[0];
  const comparable = complete;
  const leftAudit = comparable.find(a => a.id === compareLeft) ?? comparable[0];
  const rightAudit = comparable.find(a => a.id === compareRight && a.id !== leftAudit?.id) ?? comparable.find(a => a.id !== leftAudit?.id);
  const metricDelta = (key: "consensus_score" | "consensus_agreement" | "accuracy_retain" | "accuracy_forget") => leftAudit?.[key] != null && rightAudit?.[key] != null ? rightAudit[key]! - leftAudit[key]! : null;
  const signed = (v: number | null, scale = 100) => v == null ? "—" : `${v > 0 ? "+" : ""}${(v * scale).toFixed(1)}${scale === 100 ? " pp" : ""}`;
  const methodStats = ALGORITHMS.map(a => { const rows = complete.filter(x => x.algorithm === a.value); return { ...a, count: audits.filter(x => x.algorithm === a.value).length, agreement: rows.length ? rows.reduce((sum, x) => sum + (x.consensus_agreement ?? 0), 0) / rows.length : null }; });
  const maxMethodValue = Math.max(1, ...methodStats.map(x => chartMetric === "runs" ? x.count : (x.agreement ?? 0)));
  const verdicts = (["Likely Forgotten", "Not Forgotten", "Ambiguous", "Inconclusive"] as const).map(v => ({ name: v, count: complete.filter(a => (a.authoritative_verdict ?? a.rena_x_verdict ?? a.consensus_verdict) === v).length }));
  const maxVerdict = Math.max(1, ...verdicts.map(v => v.count));
  return <main className="rena-shell min-h-screen text-white">
    <aside className="rena-sidebar">
      <Link href="/" className="brand-lockup"><span className="brand-mark">R</span><span><b>RENA</b><small>UNLEARNING AUDIT LAB</small></span></Link>
      <div className="side-caption">WORKSPACE</div>
      <Link href="/dashboard" className="side-link active"><span>◫</span> Overview</Link>
      <Link href="/forgetting-audit" className="side-link font-semibold text-amber-300"><span>✦</span> LLM Forgetting Audit</Link>
      <Link href="/llm" className="side-link"><span>✧</span> LLM experiments</Link>
      <Link href="/mira" className="side-link"><span>◈</span> MIRA memory scoring</Link>
      <a href="#history" className="side-link"><span>▤</span> Classical ML history</a>
      <div className="side-bottom"><div className="status-dot"/> <span>Research environment</span><small>Audit & Evaluation Lab</small></div>
    </aside>
    <section className="rena-main">
      <header className="topbar"><div className="crumb">Workspace <span>/</span> <b>Overview</b></div><div className="top-actions"><span className="live-pill"><i/> API-connected view</span><ThemePicker/><button className="icon-btn" onClick={() => {setLoading(true); void refresh();}} aria-label="Refresh audits" title="Refresh"><span className={loading ? "spin" : ""}>↻</span></button><span className="avatar">RS</span></div></header>
      <div className="page-wrap">
        <div className="rounded-2xl border border-violet-400/30 bg-gradient-to-r from-violet-950/60 via-purple-900/30 to-black/70 p-6 shadow-xl mb-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300/30 bg-amber-400/10 px-3 py-1 text-xs font-semibold text-amber-200">
                ✦ PRIMARY RESEARCH WORKFLOW
              </span>
              <h2 className="mt-2 text-2xl font-bold text-white">RENA LLM Forgetting Audit Studio</h2>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-violet-200/80">
                Audit whether information remains accessible after a documented forgetting operation.
                Evaluate live AI models (Gemini, OpenAI, Anthropic, Ollama, Guided) with multi-angle probing,
                paired never-exposed controls, and retained-utility checks.
              </p>
            </div>
            <Link href="/forgetting-audit" className="rounded-xl border border-amber-300/40 bg-gradient-to-r from-violet-600 to-amber-600 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-violet-950/50 transition hover:brightness-110">
              Start LLM Forgetting Audit ↗
            </Link>
          </div>
        </div>

        <div className="welcome-row"><div><div className="eyebrow">RELIABLE EVALUATION FOR NEURAL UNLEARNING</div><h1>Audit workspace<span>.</span></h1><p className="subhead">Evidence-driven evaluation of model forgetting, retained utility, and verification witnesses.</p></div><Link href="/forgetting-audit" className="primary-btn"><span>✦</span> New LLM Audit</Link></div>
        {error && <div className="error-banner"><b>Connection issue</b><span>{error}. Make sure the RENA backend is running, then refresh.</span><button onClick={() => void refresh()}>Retry ↻</button></div>}
        <div className="metric-grid">
          <article className="metric-card"><div className="metric-top"><span>Total audits</span><span className="metric-icon">▤</span></div><strong>{audits.length}</strong><small>Recorded in this workspace</small><div className="metric-line"><i style={{width: `${Math.min(100, audits.length * 8)}%`}}/></div></article>
          <article className="metric-card"><div className="metric-top"><span>Completed</span><span className="metric-icon green">✓</span></div><strong>{complete.length}</strong><small>Audits with a final result</small><div className="metric-line"><i style={{width: `${audits.length ? complete.length/audits.length*100 : 0}%`}}/></div></article>
          <article className="metric-card"><div className="metric-top"><span>Mean agreement</span><span className="metric-icon violet">◈</span></div><strong>{avgAgreement == null ? "—" : pct(avgAgreement)}</strong><small>Across completed audits</small><div className="metric-line"><i style={{width: `${(avgAgreement ?? 0)*100}%`}}/></div></article>
          <article className="metric-card"><div className="metric-top"><span>Latest run</span><span className="metric-icon">◷</span></div><strong className="latest-value">{latest ? label(latest.algorithm) : "—"}</strong><small>{latest ? new Date(latest.created_at).toLocaleDateString() : "No activity yet"}</small><div className="metric-foot">{latest ? <VerdictBadge verdict={(latest.authoritative_verdict ?? latest.rena_x_verdict ?? latest.consensus_verdict)}/> : <span className="muted">Waiting for first audit</span>}</div></article>
        </div>
        <section className="insights-section" aria-label="Audit analytics">
          <div className="section-heading"><div><div className="eyebrow">LIVE WORKSPACE ANALYTICS</div><h2>Evidence at a glance</h2><p>Visual summaries are calculated from the audit records currently loaded.</p></div><span className="chart-note">{complete.length} completed runs</span></div>
          <div className="insights-grid">
            <article className="insight-card"><div className="insight-head"><div><b>Method comparison</b><small>Compare run volume or mean agreement</small></div><div className="chart-toggle" role="group" aria-label="Method chart metric"><button className={chartMetric === "runs" ? "active" : ""} onClick={() => setChartMetric("runs")}>Runs</button><button className={chartMetric === "agreement" ? "active" : ""} onClick={() => setChartMetric("agreement")}>Agreement</button></div></div>
              <div className="bar-chart">{methodStats.map(m => { const value = chartMetric === "runs" ? m.count : (m.agreement ?? 0); const display = chartMetric === "runs" ? String(m.count) : m.agreement == null ? "—" : `${(m.agreement * 100).toFixed(1)}%`; return <div className="bar-row" key={m.value}><span className="bar-label" title={m.label}>{m.short}</span><div className="bar-track"><div className="bar-fill" style={{width: `${Math.max(value > 0 ? 2 : 0, value / maxMethodValue * 100)}%`}}/></div><b>{display}</b></div>; })}</div>
              <div className="chart-foot">{chartMetric === "runs" ? "Includes all statuses" : "Completed audits only · no score means no completed data"}</div>
            </article>
            <article className="insight-card"><div className="insight-head"><div><b>Verdict distribution</b><small>Completed audits by consensus verdict</small></div><span className="chart-total">{complete.length} total</span></div>
              <div className="verdict-bars">{verdicts.map((v, i) => <button key={v.name} className="verdict-row" onClick={() => {setFilter("complete");setQuery(v.name);document.getElementById("history")?.scrollIntoView({behavior:"smooth"});}} title={`Filter history to ${v.name}`}><span className={`verdict-dot vd-${i}`}/><span className="verdict-label">{v.name}</span><span className="verdict-track"><i style={{width:`${v.count / maxVerdict * 100}%`}}/></span><b>{v.count}</b></button>)}</div>
              <div className="chart-foot">Select a verdict to filter the audit history below.</div>
            </article>
          </div>
        </section>
        <section className="compare-section" aria-label="Compare completed audits"><div className="section-heading"><div><div className="eyebrow">PAIRED REVIEW</div><h2>Compare audit runs</h2><p>Inspect two completed runs side by side. Differences are descriptive, not causal.</p></div><span className="chart-note">{comparable.length} eligible runs</span></div><div className="compare-card"><div className="compare-controls"><label>Reference run<select value={leftAudit?.id ?? ""} onChange={e=>setCompareLeft(e.target.value)} disabled={!comparable.length}><option value="" disabled>Select run</option>{comparable.map(a=><option key={a.id} value={a.id}>{label(a.algorithm)} · #{a.id.slice(0,8)}</option>)}</select></label><span className="compare-vs">VS</span><label>Comparison run<select value={rightAudit?.id ?? ""} onChange={e=>setCompareRight(e.target.value)} disabled={comparable.length<2}><option value="" disabled>Select run</option>{comparable.filter(a=>a.id!==leftAudit?.id).map(a=><option key={a.id} value={a.id}>{label(a.algorithm)} · #{a.id.slice(0,8)}</option>)}</select></label></div>{leftAudit && rightAudit ? <div className="compare-metrics"><div><small>RENA-X score</small><b>{leftAudit.rena_x_score?.toFixed(3) ?? "—"}<i>→</i>{rightAudit.rena_x_score?.toFixed(3) ?? "—"}</b><em>{signed(metricDelta("consensus_score"), 1)}</em></div><div><small>Verifier agreement</small><b>{pct(leftAudit.consensus_agreement)}<i>→</i>{pct(rightAudit.consensus_agreement)}</b><em>{signed(metricDelta("consensus_agreement"))}</em></div><div><small>Retain accuracy</small><b>{pct(leftAudit.accuracy_retain)}<i>→</i>{pct(rightAudit.accuracy_retain)}</b><em>{signed(metricDelta("accuracy_retain"))}</em></div><div><small>Forget accuracy</small><b>{pct(leftAudit.accuracy_forget)}<i>→</i>{pct(rightAudit.accuracy_forget)}</b><em>{signed(metricDelta("accuracy_forget"))}</em></div></div> : <div className="compare-empty">{comparable.length < 2 ? "Complete at least two audits to compare their results here." : "Choose two different completed runs to view the comparison."}</div>}<div className="compare-footer">{leftAudit && rightAudit ? <><span>{label(leftAudit.algorithm)} <b>→</b> {label(rightAudit.algorithm)}</span><Link href={`/audits/${rightAudit.id}`} className="open-link">Open comparison run ↗</Link></> : <span>Comparisons use currently loaded completed audit records.</span>}</div></div></section>
        <div className="mb-5 flex gap-4"><Link href="/security" className="open-link">Security & Integrity</Link><Link href="/workspace" className="open-link">Open RENA Workspace →</Link><Link href="/attack-lab" className="open-link">Open Adversarial Lab →</Link><Link href="/production" className="open-link">Production Control →</Link><Link href="/certificates" className="open-link">Verify Certificate →</Link></div><section id="launch" className="launch-section"><div className="section-heading"><div><div className="eyebrow">AUXILIARY BENCHMARK</div><h2>Classical ML unlearning methods</h2><p>Reference unlearning algorithms (Fine-Tuning, Gradient Ascent, SISA, SCRUB) for baseline comparison.</p></div><span className="step-label"><b>BASELINE</b> / METHOD SELECTION</span></div>
          <div className="launch-layout"><div className="method-grid">{ALGORITHMS.map((a,i)=><button key={a.value} disabled={running !== null || !/^\d+$/.test(splitSeed) || Number(splitSeed)>2147483647} onClick={() => setSelectedMethod(a.value)} className={`method-card ${selectedMethod===a.value?"selected":""}`}><span className="method-index">0{i+1}</span><span className="method-symbol">{a.icon}</span><b>{a.label}</b><small>{a.desc}</small><span className="method-action">{running===a.value ? <><i className="loader"/> Running…</> : <>Launch <span>↗</span></>}</span></button>)}</div>
          <aside className="config-card"><div className="config-title"><span className="config-icon">⚙</span><div><b>Audit settings</b><small>Applied to the next run</small></div></div><div className="setting-label"><span>Forget fraction</span><b>{forgetFraction}%</b></div><input aria-label="Forget fraction" type="range" min="5" max="40" step="5" value={forgetFraction} onChange={e=>setForgetFraction(Number(e.target.value))} className="range-input"/><div className="range-labels"><span>5% · Light</span><span>40% · High</span></div><div className="setting-note"><span>ⓘ</span><p>Higher fractions create a larger forget set. Use the same setting when comparing methods for a fairer experiment.</p></div><div className="setting-label mt-4"><label htmlFor="split-seed">Data split seed</label></div><input id="split-seed" aria-label="Data split seed" className="w-full rounded-lg border border-violet-300/20 bg-[#100b19] px-3 py-2 text-sm text-violet-100" type="number" min="0" max="2147483647" value={splitSeed} onChange={e=>setSplitSeed(e.target.value)} disabled={running!==null}/><div className="setting-note"><span>ⓘ</span><p>Reuse the same seed to reproduce the same stratified data split. This controls the split, not every method-specific optimizer setting.</p></div><div className="config-divider"/><div className="run-note"><span className="status-dot"/><p>{running ? `Running ${label(running)}…` : selectedMethod ? `${label(selectedMethod)} selected — ready to launch.` : "Select a method above, then launch the audit."}</p></div><button className="primary-btn" style={{width:"100%",justifyContent:"center",marginTop:12,opacity:selectedMethod?1:.45}} disabled={!selectedMethod || running!==null || !/^\d+$/.test(splitSeed) || Number(splitSeed)>2147483647} onClick={()=>selectedMethod&&void handleRunAudit(selectedMethod)}>{running?"Running audit…":"Launch selected method ↗"}</button></aside></div>
          <div className="honesty-note"><span>✧</span><p><b>Interpret results with care.</b> Audit scores are evidence-based indicators, not proof that information has been physically erased from model parameters.</p></div>
        </section>
        <section id="history" className="history-section"><div className="section-heading history-head"><div><div className="eyebrow">TRACEABLE EXPERIMENTS</div><h2>Audit history</h2><p>Review previous runs and open their evidence reports.</p></div><div className="history-actions"><button className="refresh-btn" onClick={()=>{setLoading(true);void refresh();}}>↻ Refresh</button><button className="refresh-btn export-btn" onClick={()=>{const rows=["id,algorithm,status,verdict,consensus_score,agreement,created_at",...filtered.map(a=>[a.id,a.algorithm,a.status,a.consensus_verdict??"",a.consensus_score??"",a.consensus_agreement??"",a.created_at].map(v=>`"${String(v).replaceAll("\"","\"\"")}"`).join(","))];const blob=new Blob([rows.join("\n")],{type:"text/csv;charset=utf-8"});const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download="rena-audit-history.csv";link.click();URL.revokeObjectURL(url);}}>⇩ Export CSV</button></div></div>
          <div className="history-tools"><div className="search-box"><span>⌕</span><input aria-label="Search audits" placeholder="Search method, ID, or verdict…" value={query} onChange={e=>setQuery(e.target.value)}/>{query && <button onClick={()=>setQuery("")} aria-label="Clear search">×</button>}</div><div className="filter-tabs">{["all","complete","running","pending","failed"].map(f=><button key={f} className={filter===f?"chosen":""} onClick={()=>setFilter(f)}>{f==="all"?"All":f.charAt(0).toUpperCase()+f.slice(1)}{f==="all"&&<span>{audits.length}</span>}</button>)}</div></div>
          <div className="table-wrap"><table className="audit-table"><thead><tr><th>METHOD / RUN ID</th><th>STATUS</th><th>VERDICT</th><th>RENA-X SCORE</th><th>AGREEMENT</th><th>CREATED</th><th/></tr></thead><tbody>{loading && audits.length===0 ? <tr><td colSpan={7} className="empty-cell"><span className="loader dark"/> Loading audit history…</td></tr> : filtered.length===0 ? <tr><td colSpan={7} className="empty-cell"><div className="empty-icon">⌁</div><b>{audits.length===0?"Your audit history is empty":"No matching audits"}</b><span>{audits.length===0?"Launch your first experiment above to see its results here.":"Try another search or filter."}</span></td></tr> : filtered.map(a=><tr key={a.id}><td><Link href={`/audits/${a.id}`} className="audit-name"><span className="table-method">{ALGORITHMS.find(x=>x.value===a.algorithm)?.short ?? "AI"}</span><span><b>{label(a.algorithm)}</b><small>#{a.id.slice(0,8)}</small></span></Link></td><td><span className={`status-tag ${a.status}`}><i/>{a.status}</span></td><td><VerdictBadge verdict={(a.authoritative_verdict ?? a.rena_x_verdict ?? a.consensus_verdict)}/></td><td className="number-cell">{(a.rena_x_score ?? a.consensus_score)?.toFixed(3) ?? "—"}</td><td className="number-cell">{pct(a.consensus_agreement)}</td><td className="date-cell">{new Date(a.created_at).toLocaleDateString()}<small>{new Date(a.created_at).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})}</small></td><td><Link href={`/audits/${a.id}`} className="open-link" aria-label={`Open ${label(a.algorithm)} audit`}>↗</Link></td></tr>)}</tbody></table></div><div className="table-foot"><span>Showing <b>{filtered.length}</b> of <b>{audits.length}</b> runs</span><span>Evidence is retained per audit record</span></div>
        </section>
        <footer className="page-footer"><span>RENA <i>•</i> Reliable Evaluation for Neural Unlearning Audits</span><span>Research prototype · Classical ML audits</span></footer>
      </div>
    </section>
  </main>;
}
