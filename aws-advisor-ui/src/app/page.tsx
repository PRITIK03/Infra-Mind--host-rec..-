"use client";

import { useRef, useState } from "react";
import { useJobPoller } from "@/hooks/useJobPoller";
import { StageProgress } from "@/components/StageProgress";
import { AwaitingInputPanel } from "@/components/AwaitingInputPanel";
import { ResultReport } from "@/components/ResultReport";
import { TerraformViewer } from "@/components/TerraformViewer";
import { ErrorPanel } from "@/components/ErrorPanel";
import { LiveStatsReadout } from "@/components/LiveStatsReadout";

// ── Example prompts drawn from the validated test scenarios ─────────────────
const EXAMPLES = [
  {
    label: "flash-sale e-commerce",
    prompt:
      "Flash-sale e-commerce site. We expect 50 000 registered users and up to 2 000 concurrent shoppers during sale windows. Traffic is very bursty. We need a database and probably a cache layer.",
  },
  {
    label: "steady SaaS API",
    prompt:
      "B2B SaaS REST API with about 500 registered customers. Traffic is steady, roughly 80 req/s at peak. Needs a relational database, no GPU.",
  },
  {
    label: "nightly batch processor",
    prompt:
      "Nightly CSV batch job that processes 10 GB of transaction data. Runs once at 2 AM, single concurrent job, no real-time users. No persistent database needed.",
  },
  {
    label: "ML inference service",
    prompt:
      "Real-time ML inference endpoint for image classification. About 200 req/s, low latency required, GPU acceleration needed. No database.",
  },
] as const;

// ── Wordmark ─────────────────────────────────────────────────────────────────
function Wordmark() {
  return (
    <div className="flex items-baseline gap-2">
      <span className="font-mono text-sm font-medium text-amber tracking-tight">
        aws-instance-advisor
      </span>
      <span className="font-mono text-xs text-ink-dim">v2</span>
    </div>
  );
}

// ── Command-prompt input ──────────────────────────────────────────────────────
interface InputPanelProps {
  onSubmit: (msg: string) => void;
  disabled: boolean;
}

function InputPanel({ onSubmit, disabled }: InputPanelProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl+Enter submits; plain Enter adds newline
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      const trimmed = value.trim();
      if (trimmed && !disabled) {
        onSubmit(trimmed);
        setValue("");
      }
    }
  };

  const fillExample = (prompt: string) => {
    setValue(prompt);
    textareaRef.current?.focus();
  };

  return (
    <div className="w-full">
      {/* Prompt chips */}
      <div className="mb-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="font-mono text-xs text-ink-dim uppercase tracking-widest">
            examples
          </span>
          <span className="flex-1 border-t border-border-subtle" />
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => fillExample(ex.prompt)}
              disabled={disabled}
              className={[
                "border border-border-subtle px-3 py-1.5",
                "font-mono text-xs text-ink-dim",
                "transition-colors duration-150",
                "hover:text-amber hover:border-amber/40",
                "disabled:opacity-30 disabled:cursor-not-allowed",
              ].join(" ")}
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main input */}
      <form onSubmit={handleSubmit}>
        <div className="flex gap-0 border border-border-subtle">
          {/* Prompt sigil */}
          <div className="flex items-start pt-3 px-3 shrink-0 border-r border-border-subtle">
            <span className="font-mono text-sm text-amber leading-6">&gt;</span>
          </div>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="describe your application and workload…"
            disabled={disabled}
            rows={4}
            className={[
              "flex-1 bg-transparent resize-none",
              "font-sans text-sm text-ink placeholder:text-ink-dim",
              "px-3 py-3 outline-none leading-relaxed",
              "disabled:opacity-40 disabled:cursor-not-allowed",
            ].join(" ")}
          />
        </div>

        <div className="flex items-center justify-between mt-2">
          <span className="font-mono text-xs text-ink-dim">
            ⌘↵ to submit
          </span>
          <button
            type="submit"
            disabled={!value.trim() || disabled}
            className={[
              "border border-border-subtle px-5 py-2",
              "font-mono text-xs text-ink-muted uppercase tracking-wider",
              "transition-colors duration-150",
              "hover:text-amber hover:border-amber/40",
              "disabled:opacity-30 disabled:cursor-not-allowed",
              "disabled:hover:text-ink-muted disabled:hover:border-border-subtle",
            ].join(" ")}
          >
            run →
          </button>
        </div>
      </form>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Home() {
  const {
    status,
    currentStage,
    nextQuestion,
    jobResponse,
    error,
    isActive,
    submit,
    answer,
    reset,
  } = useJobPoller();

  const showLanding = status === null;
  const showProgress = status !== null && status !== "done" && status !== "error";
  const showResult = status === "done" && jobResponse?.result != null;
  const showError = status === "error" && error != null;

  const sdr = jobResponse?.result?.system_design_recommendation;
  const tfFiles = jobResponse?.result?.terraform_files;
  const technicalNeeds = jobResponse?.result?.technical_needs;
  const instanceCandidates = jobResponse?.result?.instance_candidates;

  return (
    <div className="min-h-dvh flex flex-col">
      {/* ── Top bar ── */}
      <header className="border-b border-border-subtle px-6 py-3 flex items-center justify-between shrink-0">
        <Wordmark />
        <span className="font-mono text-xs text-ink-dim hidden sm:block">
          {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}
        </span>
      </header>

      {/* ── Main content ── */}
      <main className="flex-1 w-full max-w-3xl mx-auto px-6 py-12 flex flex-col gap-10">

        {/* ── Landing / input ── */}
        {showLanding && (
          <section className="animate-fade-in">
            <div className="mb-8">
              <h1 className="font-mono text-xl font-medium text-ink mb-2 tracking-tight">
                AWS Instance Advisor
              </h1>
              <p className="font-sans text-sm text-ink-muted leading-relaxed max-w-xl">
                Describe your application and workload. The advisor reasons about
                your requirements, researches live AWS instance data, and returns
                a vetted compute, database, cache, and load-balancer recommendation
                — plus ready-to-apply Terraform.
              </p>
              <div className="mt-4">
                <LiveStatsReadout />
              </div>
            </div>
            <InputPanel onSubmit={submit} disabled={isActive} />
          </section>
        )}

        {/* ── Active job: re-prompt input at top + progress below ── */}
        {showProgress && (
          <section className="animate-fade-in space-y-8">
            {/* Allow new input only when awaiting — blocked while running */}
            {status === "awaiting_input" && nextQuestion ? (
              <AwaitingInputPanel
                question={nextQuestion}
                onAnswer={answer}
                disabled={false}
              />
            ) : (
              /* Compact "running" header */
              <div className="flex items-center gap-3">
                <span className="w-2 h-2 rounded-full bg-amber animate-pulse shrink-0" />
                <span className="font-mono text-sm text-ink-muted">
                  {currentStage || "Initializing"}
                </span>
              </div>
            )}

            <hr className="console-rule" />

            <StageProgress
              currentStage={currentStage}
              status={status}
            />
          </section>
        )}

        {/* ── Result ── */}
        {showResult && sdr && (
          <section>
            {/* "New analysis" action in top bar */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-amber uppercase tracking-widest">
                  recommendation
                </span>
                <span className="flex-1 border-t border-amber/20 w-8" />
              </div>
              <button
                onClick={reset}
                className={[
                  "border border-border-subtle px-3 py-1.5",
                  "font-mono text-xs text-ink-dim uppercase tracking-wider",
                  "transition-colors duration-150",
                  "hover:text-amber hover:border-amber/40",
                ].join(" ")}
              >
                ↩ new analysis
              </button>
            </div>

            <ResultReport sdr={sdr} technicalNeeds={technicalNeeds} instanceCandidates={instanceCandidates} />

            {tfFiles && Object.keys(tfFiles).length > 0 && (
              <>
                <hr className="console-rule my-0" />
                <div className="py-6">
                  <TerraformViewer files={tfFiles} />
                </div>
              </>
            )}
          </section>
        )}

        {/* ── Error ── */}
        {showError && (
          <section>
            <ErrorPanel message={error!} onRetry={reset} />
          </section>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-border-subtle px-6 py-3 shrink-0">
        <p className="font-mono text-xs text-ink-dim text-center">
          AI-generated infrastructure — review every{" "}
          <code className="text-ink-dim">terraform plan</code> before applying
        </p>
      </footer>
    </div>
  );
}
