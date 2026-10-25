// Mirrors backend/app/schemas/audit.py exactly -- keep these in sync.

export type Algorithm = "fine_tuning" | "gradient_ascent" | "ground_truth_retraining" | "sisa" | "prism" | "echo" | "neggrad" | "salun" | "unsir" | "scrub";
export type Verdict = "Likely Forgotten" | "Not Forgotten" | "Ambiguous" | "Inconclusive";

export interface ModuleResult {
  module_name: string;
  raw_statistic: number;
  p_value: number | null;
  forget_score: number;
  reliability: number;
  reliability_reason: string;
  detail: Record<string, unknown>;
}

export interface ConsensusContribution {
  module: string;
  forget_score: number;
  reliability: number;
  weight_pct: number;
}

export interface Diagnosis {
  verdict: Verdict;
  findings: string[];
  recommendations: string[];
  advanced_audit?: {
    verdict: string;
    worst_case_score: number;
    certification_score?: number;
    axes: Record<string, number | string>;
    interpretation: string;
    evidence_profile?: Record<string, number>;
    reference_alignment?: { calibrated: boolean; reference_count: number; reference_runtime_median_seconds?: number };
  };
}

export interface Audit {
  id: string;
  algorithm: Algorithm;
  status: "pending" | "running" | "complete" | "failed";
  runtime_seconds: number | null;
  sisa_shards_retrained: number | null;
  accuracy_retain: number | null;
  accuracy_test: number | null;
  accuracy_forget: number | null;
  accuracy_original_test: number | null;
  consensus_verdict: Verdict | null;
  consensus_score: number | null;
  consensus_agreement: number | null;
  rena_x_verdict: Verdict | null;
  rena_x_score: number | null;
  authoritative_verdict: Verdict | null;
  module_results: ModuleResult[] | null;
  method_trace: Record<string, unknown> | null;
  diagnosis: Diagnosis | null;
  dataset_summary?: Record<string, unknown> | null;
  reference_calibration?: { enabled: boolean; reference_count: number; reference_seed: number; reference_runtime_median_seconds?: number; calibration?: Record<string, unknown> | null };
  created_at: string;
  completed_at: string | null;
}

export const VERDICT_COLOR: Record<Verdict, string> = {
  "Likely Forgotten": "#15803D",
  "Not Forgotten": "#B91C1C",
  "Ambiguous": "#B45309",
  "Inconclusive": "#6B7280",
};

export type LLMMethod =
  | "retain_only_retraining"
  | "retain_finetuning"
  | "gradient_ascent"
  | "gradient_difference"
  | "npo";

export type LLMDevice = "auto" | "cpu" | "cuda" | "mps";

export interface LLMJobConfig {
  model: string;
  methods: LLMMethod[];
  seed: number;
  epochs: number;
  warmup_epochs: number;
  learning_rate: number;
  max_length: number;
  max_new_tokens: number;
  context_sizes: number[];
  device: LLMDevice;
  full_niah: boolean;
}

export interface LLMJob {
  id: string;
  owner_id: string;
  status: "queued" | "running" | "complete" | "failed" | "interrupted";
  created_at: string;
  config: LLMJobConfig;
  output_dir: string;
  command: string[] | null;
  pid?: number;
  return_code?: number;
  completed_at?: string | null;
  message?: string;
  log_tail: string;
}

export interface LLMJobResultsFile {
  path: string;
  data?: Record<string, unknown>;
  unavailable?: boolean;
}

export interface LLMJobResults {
  job: LLMJob;
  files: LLMJobResultsFile[];
}

export interface MIRAItem {
  id: string;
  rank: number;
  score: number;
  signals: Record<string, unknown>;
  components: Record<string, unknown>;
}

export interface MIRAScoreResult {
  method: string;
  items: MIRAItem[];
  weights: Record<string, number>;
  limitations?: string[];
}
