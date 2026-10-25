"use client";
import Link from "next/link";
import { useState } from "react";
import ThemePicker from "@/components/ThemePicker";
import { scoreMIRA, MIRAScoreResult } from "@/lib/api";

const starter = [
  { id: "memory-A", semantic_relevance: 0.9, historical_usefulness: 0.8, query_context_usefulness: 0.9, redundancy: 0.1, confidence: 0.95 },
  { id: "memory-B", semantic_relevance: 0.5, historical_usefulness: 0.4, query_context_usefulness: 0.3, redundancy: 0.7, confidence: 0.8 },
  { id: "memory-C", semantic_relevance: 0.7, historical_usefulness: 0.9, query_context_usefulness: 0.6, redundancy: 0.2, confidence: 0.75 },
];
const inputClass = "w-full rounded-xl border border-violet-300/15 bg-[#100b19] px-3 py-2.5 text-xs text-violet-50 outline-none focus:border-violet-400";
export default function MIRAPage() {
  const [raw, setRaw] = useState(JSON.stringify(starter, null, 2));
  const [result, setResult] = useState<MIRAScoreResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function run() {
    setError(""); setResult(null);
    let parsed: unknown;
    try { parsed = JSON.parse(raw); } catch { setError("Please enter valid JSON."); return; }
    if (!Array.isArray(parsed) || !parsed.length || parsed.length > 100) { setError("Provide an array of 1–100 memory records."); return; }
    setBusy(true);
    try { setResult(await scoreMIRA(parsed as Record<string, unknown>[])); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <main className="rena-shell min-h-screen text-white"><aside className="rena-sidebar"><Link href="/" className="brand-lockup"><span className="brand-mark">R</span><span><b>RENA</b><small>UNLEARNING AUDIT LAB</small></span></Link><div className="side-caption">WORKSPACE</div><Link href="/dashboard" className="side-link"><span>◫</span> Overview</Link><Link href="/llm" className="side-link"><span>✧</span> LLM experiments</Link><Link href="/mira" className="side-link active"><span>◈</span> MIRA memory scoring</Link><div className="side-bottom"><div className="status-dot"/><span>Research utility</span><small>Transparent prototype scoring</small></div></aside>
    <section className="rena-main"><header className="topbar"><div className="crumb">Workspace <span>/</span> <b>MIRA memory scoring</b></div><div className="top-actions"><ThemePicker/><Link href="/dashboard" className="refresh-btn" style={{width:"auto",padding:"0 10px",display:"grid",placeItems:"center",textDecoration:"none"}}>Classical ML ↗</Link></div></header>
      <div className="page-wrap"><div className="welcome-row"><div><div className="eyebrow">MEMORY IMPORTANCE & RELEVANCE ADAPTATION</div><h1>MIRA scoring lab<span>.</span></h1><p className="subhead">Rank memory records using explicit, inspectable importance and relevance signals.</p></div><span className="chart-note">Prototype · caller-supplied signals</span></div>
        <section className="insight-card mb-5"><div className="section-heading"><div><div className="eyebrow">SIGNAL INPUT</div><h2>Memory records</h2><p>Enter an array of records. Each signal must be a number from 0 to 1; redundancy is a penalty.</p></div></div><textarea className={`${inputClass} min-h-[300px] font-mono`} value={raw} onChange={e=>setRaw(e.target.value)} spellCheck={false} aria-label="MIRA memory records JSON"/><div className="mt-3 flex flex-wrap items-center justify-between gap-3"><span className="text-[10px] text-violet-200/55">Required: id, semantic_relevance, historical_usefulness, query_context_usefulness, redundancy, confidence</span><button className="rounded-xl border border-violet-300/30 bg-gradient-to-r from-violet-600 to-purple-500 px-5 py-3 text-xs font-semibold text-white disabled:opacity-40" onClick={()=>void run()} disabled={busy}>{busy?"Scoring…":"Calculate MIRA ranking"}</button></div>{error&&<div className="error-banner mt-3"><b>Input or API error</b><span>{error}</span></div>}</section>
        {result&&<section className="insight-card mb-5"><div className="section-heading"><div><div className="eyebrow">RANKED OUTPUT</div><h2>Importance estimates</h2><p>{result.method} · higher score means higher estimated importance under the displayed prototype weights.</p></div></div><div className="table-wrap"><table className="audit-table"><thead><tr><th>RANK</th><th>MEMORY ID</th><th>SCORE</th><th>COMPONENTS</th></tr></thead><tbody>{result.items.map(item=><tr key={item.id}><td>{item.rank}</td><td><b>{item.id}</b></td><td className="number-cell">{item.score.toFixed(3)}</td><td><details><summary className="cursor-pointer text-violet-300">Inspect signals</summary><pre className="mt-2 max-w-xl overflow-auto whitespace-pre-wrap text-[10px] text-violet-100/70">{JSON.stringify({signals:item.signals,weighted_components:item.components},null,2)}</pre></details></td></tr>)}</tbody></table></div><h3 className="mt-5 mb-2 text-xs font-semibold text-violet-200">Weights</h3><div className="flex flex-wrap gap-2">{Object.entries(result.weights).map(([key,value])=><span key={key} className="rounded-lg border border-violet-300/15 px-2 py-1 text-[10px] text-violet-200">{key}: {(value*100).toFixed(0)}%</span>)}</div></section>}
        <section className="insight-card"><div className="eyebrow">SCIENTIFIC LIMITATIONS</div><h2 className="mt-2 mb-3 text-sm font-semibold">Interpretation guidance</h2><ul className="grid gap-2 text-xs leading-relaxed text-violet-100/65">{(result?.limitations ?? ["Signal values are supplied by the user and are not inferred or independently validated.","Weights are transparent prototype assumptions, not empirically calibrated parameters.","A memory importance ranking is not proof of forgetting, and must not override privacy or deletion requirements."]).map(x=><li key={x}>• {x}</li>)}</ul></section><footer className="page-footer"><span>RENA · Reliable Evaluation for Neural Unlearning Audits</span><span>MIRA is decision support, not a forgetting verifier</span></footer></div></section></main>;
}
