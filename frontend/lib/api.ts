export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

import type {
  Algorithm,
  Verdict,
  ModuleResult,
  Audit,
  LLMMethod,
  LLMDevice,
  LLMJobConfig,
  LLMJob,
  LLMJobResultsFile,
  LLMJobResults,
  MIRAItem,
  MIRAScoreResult,
  ForgettingAuditSummary,
  UnifiedAuditRow,
} from "@/lib/types";

export type {
  Algorithm,
  Verdict,
  ModuleResult,
  Audit,
  LLMMethod,
  LLMDevice,
  LLMJobConfig,
  LLMJob,
  LLMJobResultsFile,
  LLMJobResults,
  MIRAItem,
  MIRAScoreResult,
  ForgettingAuditSummary,
  UnifiedAuditRow,
} from "@/lib/types";

export function authHeaders(extra: Record<string, string> = {}) {
  const token =
    typeof window !== "undefined"
      ? localStorage.getItem("rena_token")
      : null;

  return {
    ...extra,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token =
    typeof window !== "undefined"
      ? localStorage.getItem("rena_token")
      : null;

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(API + path, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!res.ok) {
    let msg = `Request failed (${res.status})`;

    try {
      const data = await res.json();
      if (typeof data?.detail === "string") {
        msg = data.detail;
      }
    } catch {
      // Keep the HTTP status message.
    }

    throw new Error(msg);
  }

  return res;
}

async function apiJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await apiFetch(path, init);
  return res.json() as Promise<T>;
}

export async function auth(
  mode: "login" | "register",
  email: string,
  password: string,
) {
  const res = await fetch(`${API}/production/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
    cache: "no-store",
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.detail || "Authentication failed");
  }

  if (typeof window !== "undefined") {
    localStorage.setItem("rena_token", data.token);
    document.cookie = `rena_token=${encodeURIComponent(
      data.token,
    )}; Path=/; SameSite=Lax`;
  }

  return data;
}

export function logout() {
  if (typeof window !== "undefined") {
    localStorage.removeItem("rena_token");
    document.cookie =
      "rena_token=; Path=/; Max-Age=0; SameSite=Lax";
  }
}

export async function me() {
  return apiJson("/production/auth/me");
}

export async function listForgettingAudits(): Promise<ForgettingAuditSummary[]> {
  return apiJson<ForgettingAuditSummary[]>("/forgetting-audits");
}

export async function listAudits(limit = 50): Promise<Audit[]> {
  return apiJson<Audit[]>(`/audits?limit=${limit}`);
}

export async function createAudit(
  algorithm: Algorithm,
  forgetFraction = 0.1,
  splitSeed = 42,
): Promise<Audit> {
  return apiJson<Audit>("/audits", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      algorithm,
      forget_fraction: forgetFraction,
      split_seed: splitSeed,
      reference_retrains: 10,
      reference_seed: 50000,
      project_id: null,
    }),
  });
}

export async function getAudit(auditId: string): Promise<Audit> {
  return apiJson<Audit>(
    `/audits/${encodeURIComponent(auditId)}`,
  );
}

export async function createLLMJob(
  config: LLMJobConfig,
): Promise<LLMJob> {
  return apiJson<LLMJob>("/llm/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: config.model,
      methods: config.methods,
      seed: config.seed ?? 42,
      epochs: config.epochs ?? 1,
      warmup_epochs: config.warmup_epochs ?? 1,
      learning_rate: config.learning_rate ?? 2e-5,
      max_length: config.max_length ?? 256,
      max_new_tokens: config.max_new_tokens ?? 32,
      context_sizes: config.context_sizes ?? [8, 32],
      device: config.device ?? "auto",
      full_niah: config.full_niah ?? false,
    }),
  });
}

export async function listLLMJobs(): Promise<LLMJob[]> {
  return apiJson<LLMJob[]>("/llm/jobs");
}

export async function getLLMJob(
  jobId: string,
): Promise<LLMJob> {
  return apiJson<LLMJob>(
    `/llm/jobs/${encodeURIComponent(jobId)}`,
  );
}

export async function getLLMJobResults(
  jobId: string,
): Promise<LLMJobResults> {
  return apiJson<LLMJobResults>(
    `/llm/jobs/${encodeURIComponent(jobId)}/results`,
  );
}

export async function scoreMIRA(
  items: Record<string, unknown>[],
): Promise<MIRAScoreResult> {
  return apiJson<MIRAScoreResult>("/research/mira/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
}
