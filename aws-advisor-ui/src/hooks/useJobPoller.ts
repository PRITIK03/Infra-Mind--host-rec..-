"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { answerJob, createJob, pollJob } from "@/lib/api";
import type { JobResponse, JobStatus } from "@/lib/types";

const POLL_INTERVAL_MS = 2500;

export interface UseJobPollerReturn {
  status: JobStatus | null;
  currentStage: string;
  nextQuestion: string | null;
  jobResponse: JobResponse | null;
  error: string | null;
  isActive: boolean;
  submit: (message: string) => Promise<void>;
  answer: (reply: string) => Promise<void>;
  reset: () => void;
}

export function useJobPoller(): UseJobPollerReturn {
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobResponse, setJobResponse] = useState<JobResponse | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [currentStage, setCurrentStage] = useState<string>("");
  const [nextQuestion, setNextQuestion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Track whether we should keep polling
  const pollingRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    pollingRef.current = false;
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const applyResponse = useCallback(
    (resp: JobResponse) => {
      setJobResponse(resp);
      setStatus(resp.status);
      setCurrentStage(resp.current_stage);

      if (resp.status === "awaiting_input") {
        setNextQuestion(resp.next_question ?? null);
        stopPolling();
      } else if (resp.status === "done" || resp.status === "error") {
        if (resp.error) setError(resp.error);
        stopPolling();
      }
    },
    [stopPolling]
  );

  const startPolling = useCallback(
    (id: string) => {
      pollingRef.current = true;
      intervalRef.current = setInterval(async () => {
        if (!pollingRef.current) return;
        try {
          const resp = await pollJob(id);
          applyResponse(resp);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Polling failed");
          stopPolling();
        }
      }, POLL_INTERVAL_MS);
    },
    [applyResponse, stopPolling]
  );

  // Clean up on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  const submit = useCallback(
    async (message: string) => {
      stopPolling();
      setJobId(null);
      setJobResponse(null);
      setStatus(null);
      setCurrentStage("Initializing");
      setNextQuestion(null);
      setError(null);

      try {
        const { job_id } = await createJob(message);
        setJobId(job_id);
        setStatus("collecting");
        startPolling(job_id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to start job");
        setStatus("error");
      }
    },
    [stopPolling, startPolling]
  );

  const answer = useCallback(
    async (reply: string) => {
      if (!jobId) return;
      setNextQuestion(null);
      setStatus("running");
      setError(null);

      try {
        await answerJob(jobId, reply);
        startPolling(jobId);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to submit answer");
        setStatus("error");
      }
    },
    [jobId, startPolling]
  );

  const reset = useCallback(() => {
    stopPolling();
    setJobId(null);
    setJobResponse(null);
    setStatus(null);
    setCurrentStage("");
    setNextQuestion(null);
    setError(null);
  }, [stopPolling]);

  const isActive =
    status === "collecting" || status === "running";

  return {
    status,
    currentStage,
    nextQuestion,
    jobResponse,
    error,
    isActive,
    submit,
    answer,
    reset,
  };
}
