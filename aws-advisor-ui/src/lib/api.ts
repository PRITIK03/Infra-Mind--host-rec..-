import type { JobResponse } from "./types";

const BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore parse error — use status code message
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

/** POST /api/recommend — kick off a new job */
export async function createJob(message: string): Promise<{ job_id: string }> {
  return request<{ job_id: string }>("/api/recommend", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

/** GET /api/recommend/{job_id} — poll job status */
export async function pollJob(jobId: string): Promise<JobResponse> {
  return request<JobResponse>(`/api/recommend/${jobId}`);
}

/** POST /api/recommend/{job_id}/answer — submit a follow-up answer */
export async function answerJob(
  jobId: string,
  answer: string
): Promise<{ job_id: string; status: string }> {
  return request(`/api/recommend/${jobId}/answer`, {
    method: "POST",
    body: JSON.stringify({ answer }),
  });
}

/** GET /api/stats — live instance type counts for landing page readout */
export async function getStats(): Promise<{ ec2: number; rds: number; cache: number }> {
  return request<{ ec2: number; rds: number; cache: number }>("/api/stats");
}
