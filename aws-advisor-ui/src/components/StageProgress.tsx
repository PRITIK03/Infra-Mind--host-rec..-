"use client";

import { ORDERED_STAGES } from "@/lib/types";
import type { JobStatus } from "@/lib/types";

interface Props {
  currentStage: string;
  status: JobStatus;
}

export function StageProgress({ currentStage, status }: Props) {
  const currentIdx = ORDERED_STAGES.findIndex((s) => s === currentStage);
  // If we don't find an exact match, treat as "in progress somewhere"
  const effectiveIdx = currentIdx === -1 ? 0 : currentIdx;

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-4">
        <span className="font-mono text-xs text-ink-muted uppercase tracking-widest">
          execution trace
        </span>
        <span className="flex-1 border-t border-border-subtle" />
        {(status === "collecting" || status === "running") && (
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-amber animate-pulse" />
            <span className="font-mono text-xs text-amber">running</span>
          </span>
        )}
        {status === "awaiting_input" && (
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-ink-muted" />
            <span className="font-mono text-xs text-ink-muted">waiting</span>
          </span>
        )}
      </div>

      <ol className="space-y-0">
        {ORDERED_STAGES.map((stage, idx) => {
          const isDone = idx < effectiveIdx;
          const isCurrent =
            idx === effectiveIdx &&
            (status === "collecting" ||
              status === "running" ||
              status === "awaiting_input");
          const isPending = idx > effectiveIdx;

          return (
            <li key={stage} className="flex items-start gap-3 py-1.5">
              {/* Step indicator */}
              <div className="flex flex-col items-center mt-0.5 shrink-0">
                <div
                  className={[
                    "w-4 h-4 flex items-center justify-center shrink-0",
                    isDone
                      ? "text-amber"
                      : isCurrent
                      ? "text-amber"
                      : "text-ink-dim",
                  ].join(" ")}
                >
                  {isDone ? (
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 14 14"
                      fill="none"
                      aria-hidden
                    >
                      <rect
                        x="0.5"
                        y="0.5"
                        width="13"
                        height="13"
                        rx="1.5"
                        stroke="currentColor"
                        strokeOpacity="0.6"
                      />
                      <path
                        d="M3.5 7L6 9.5L10.5 5"
                        stroke="currentColor"
                        strokeWidth="1.25"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  ) : isCurrent ? (
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 14 14"
                      fill="none"
                      aria-hidden
                    >
                      <rect
                        x="0.5"
                        y="0.5"
                        width="13"
                        height="13"
                        rx="1.5"
                        stroke="currentColor"
                      />
                      <rect
                        x="4"
                        y="4"
                        width="6"
                        height="6"
                        rx="0.75"
                        fill="currentColor"
                      />
                    </svg>
                  ) : (
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 14 14"
                      fill="none"
                      aria-hidden
                    >
                      <rect
                        x="0.5"
                        y="0.5"
                        width="13"
                        height="13"
                        rx="1.5"
                        stroke="currentColor"
                        strokeOpacity="0.25"
                      />
                    </svg>
                  )}
                </div>
                {/* Connector line */}
                {idx < ORDERED_STAGES.length - 1 && (
                  <div
                    className={[
                      "w-px flex-1 min-h-[8px] mt-0.5",
                      isDone
                        ? "bg-amber/30"
                        : "bg-border-subtle",
                    ].join(" ")}
                  />
                )}
              </div>

              {/* Stage label */}
              <span
                className={[
                  "font-mono text-sm leading-tight pb-1.5",
                  isDone
                    ? "text-ink/50"
                    : isCurrent
                    ? "text-ink"
                    : isPending
                    ? "text-ink-dim"
                    : "text-ink-muted",
                ].join(" ")}
              >
                {stage}
                {isCurrent && (
                  <span className="inline-block ml-1.5 text-amber animate-pulse">
                    ···
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
