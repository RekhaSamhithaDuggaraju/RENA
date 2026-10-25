import Link from "next/link";
const checks=[
 ["Authentication","Bearer-token authentication is required for workspace, audit, project and certificate operations."],
 ["Authorization","Project and audit resources enforce owner access before operations."],
 ["Input validation","Asset kinds and extensions are allowlisted and upload paths are contained."],
 ["File integrity","Uploaded assets record SHA-256 provenance metadata."],
 ["Certificate integrity","Canonical certificates use SHA-256 plus an HMAC signature."],
 ["Audit trail","Certificate issuance and workspace activity create auditable events."],
 ["Public verification","Certificate verification is intentionally public; private audit evidence remains protected."],
];
export default function Security(){return <main className="min-h-screen bg-[#0c0815] text-white p-8"><div className="max-w-5xl mx-auto"><Link href="/dashboard" className="text-[#bda6ff]">← Dashboard</Link><div className="mt-8"><div className="text-xs uppercase tracking-[.25em] text-[#bda6ff]">TRUST & INTEGRITY</div><h1 className="text-4xl font-semibold mt-2">Security &amp; Integrity</h1><p className="text-white/60 mt-2 max-w-3xl">RENA treats verification evidence as a protected project resource. Public certificate verification is the deliberate exception.</p></div><div className="grid md:grid-cols-2 gap-4 mt-8">{checks.map(([title,desc])=><section key={title} className="rounded-2xl border border-white/10 bg-[#15101f] p-5"><div className="flex items-center justify-between gap-4"><h2 className="font-semibold">{title}</h2><span className="text-emerald-300">✓</span></div><p className="text-sm text-white/55 mt-2">{desc}</p></section>)}</div><section className="rounded-2xl border border-white/10 bg-[#15101f] p-6 mt-5"><h2 className="text-xl font-semibold">Scope</h2><p className="text-white/55 mt-2">Security controls protect application resources and certificate integrity. They do not make the underlying statistical audit a proof of physical parameter deletion or universal unlearning.</p></section></div></main>}
