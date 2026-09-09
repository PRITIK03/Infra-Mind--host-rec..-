// ─── API contract types ────────────────────────────────────────────────────
// Mirror the FastAPI backend schemas exactly. Field names match
// the JSON produced by _job_response() and _serialize_result().

export type JobStatus =
  | "collecting"
  | "awaiting_input"
  | "running"
  | "done"
  | "error";

export type CacheEngine = "Memcached" | "Redis" | "Valkey";

export interface InstanceRecommendation {
  recommended_instance: string;
  why: string;
  assumptions: string[];
  confidence: "low" | "medium" | "high";
  alternative_instance?: string | null;
  trade_off?: string | null;
}

export interface DatabaseRecommendation {
  needed: boolean;
  recommended_instance?: string | null;
  engine_suggestion?: string | null;
  why: string;
  assumptions: string[];
  confidence: "low" | "medium" | "high";
  alternative_instance?: string | null;
}

export interface CacheRecommendation {
  needed: boolean;
  recommended_instance?: string | null;
  engine?: CacheEngine | null;
  why: string;
  assumptions: string[];
  confidence: "low" | "medium" | "high";
  alternative_instance?: string | null;
  alternative_engine?: CacheEngine | null;
}

export interface LoadBalancerRecommendation {
  needed: boolean;
  load_balancer_type?: string | null;
  why: string;
}

export interface SystemDesignRecommendation {
  compute: InstanceRecommendation;
  database: DatabaseRecommendation;
  cache: CacheRecommendation;
  load_balancer: LoadBalancerRecommendation;
  architecture_summary: string;
}

export interface JobResult {
  system_design_recommendation?: SystemDesignRecommendation;
  /** Legacy V1 fallback — rendered if system_design_recommendation is absent */
  recommendation?: InstanceRecommendation;
  terraform_files?: Record<string, string>;
}

// ─── Polling response shapes ───────────────────────────────────────────────
// Fields are conditionally present — absent when not applicable.

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  current_stage: string;
  created_at: number;
  // awaiting_input only:
  next_question?: string;
  // done only:
  result?: JobResult;
  // error only:
  error?: string;
}

// ─── Ordered stage list — drives the progress step-list ───────────────────
// Matches STAGE_LABELS in app/api/jobs.py.

export const ORDERED_STAGES = [
  "Initializing",
  "Collecting requirements",
  "Validating requirements",
  "Reasoning about system design",
  "Researching compute options",
  "Researching database options",
  "Researching cache options",
  "Building final recommendation",
  "Generating Terraform",
] as const;

export type StageName = (typeof ORDERED_STAGES)[number];
