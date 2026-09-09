"use client";

import type { SystemDesignRecommendation } from "@/lib/types";
import { ArchSummary } from "./ArchSummary";
import { ConfidenceDot } from "./ConfidenceDot";

interface Props {
  sdr: SystemDesignRecommendation;
}

// ── Tiny helpers ────────────────────────────────────────────────────────────

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="font-mono text-xs text-ink-muted uppercase tracking-widest">
        {label}
      </span>
      <span className="flex-1 border-t border-border-subtle" />
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-xs text-ink-dim uppercase tracking-wider">
        {label}
      </span>
      <div className="text-ink">{children}</div>
    </div>
  );
}

function MonoValue({ value }: { value: string }) {
  return (
    <span className="font-mono text-sm text-ink">{value}</span>
  );
}

function AssumptionsList({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <details className="group mt-1">
      <summary className="inline-flex items-center gap-1.5 font-mono text-xs text-ink-dim hover:text-ink-muted transition-colors cursor-pointer select-none">
        <span className="group-open:hidden">▶</span>
        <span className="hidden group-open:inline">▼</span>
        {items.length} assumption{items.length !== 1 ? "s" : ""}
      </summary>
      <ul className="mt-2 space-y-1 pl-4 border-l border-border-subtle">
        {items.map((a, i) => (
          <li key={i} className="font-sans text-xs text-ink-muted leading-relaxed">
            {a}
          </li>
        ))}
      </ul>
    </details>
  );
}

// ── Tier sections ────────────────────────────────────────────────────────────

function ComputeSection({
  compute,
}: {
  compute: SystemDesignRecommendation["compute"];
}) {
  return (
    <section>
      <SectionHeader label="compute" />
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="instance type">
          <MonoValue value={compute.recommended_instance} />
        </Field>
        <Field label="confidence">
          <ConfidenceDot level={compute.confidence} />
        </Field>
        <Field label="rationale">
          <p className="font-sans text-sm text-ink-muted leading-relaxed">
            {compute.why}
          </p>
        </Field>
        {compute.alternative_instance && (
          <Field label="alternative">
            <MonoValue value={compute.alternative_instance} />
            {compute.trade_off && (
              <p className="font-sans text-xs text-ink-dim mt-0.5 leading-relaxed">
                {compute.trade_off}
              </p>
            )}
          </Field>
        )}
      </div>
      <AssumptionsList items={compute.assumptions} />
    </section>
  );
}

function DatabaseSection({
  database,
}: {
  database: SystemDesignRecommendation["database"];
}) {
  if (!database.needed) {
    return (
      <section>
        <SectionHeader label="database" />
        <p className="font-sans text-sm text-ink-muted">{database.why}</p>
      </section>
    );
  }
  return (
    <section>
      <SectionHeader label="database" />
      <div className="grid gap-3 sm:grid-cols-2">
        {database.recommended_instance && (
          <Field label="instance class">
            <MonoValue value={database.recommended_instance} />
          </Field>
        )}
        {database.engine_suggestion && (
          <Field label="engine">
            <MonoValue value={database.engine_suggestion} />
          </Field>
        )}
        <Field label="confidence">
          <ConfidenceDot level={database.confidence} />
        </Field>
        <Field label="rationale">
          <p className="font-sans text-sm text-ink-muted leading-relaxed">
            {database.why}
          </p>
        </Field>
        {database.alternative_instance && (
          <Field label="alternative">
            <MonoValue value={database.alternative_instance} />
          </Field>
        )}
      </div>
      <AssumptionsList items={database.assumptions} />
    </section>
  );
}

function CacheSection({
  cache,
}: {
  cache: SystemDesignRecommendation["cache"];
}) {
  if (!cache.needed) {
    return (
      <section>
        <SectionHeader label="cache" />
        <p className="font-sans text-sm text-ink-muted">{cache.why}</p>
      </section>
    );
  }
  return (
    <section>
      <SectionHeader label="cache" />
      <div className="grid gap-3 sm:grid-cols-2">
        {cache.recommended_instance && (
          <Field label="node type">
            <MonoValue value={cache.recommended_instance} />
          </Field>
        )}
        {cache.engine && (
          <Field label="engine">
            <MonoValue value={cache.engine} />
          </Field>
        )}
        <Field label="confidence">
          <ConfidenceDot level={cache.confidence} />
        </Field>
        <Field label="rationale">
          <p className="font-sans text-sm text-ink-muted leading-relaxed">
            {cache.why}
          </p>
        </Field>
        {cache.alternative_instance && (
          <Field label="alternative">
            <MonoValue value={cache.alternative_instance} />
            {cache.alternative_engine && (
              <span className="font-mono text-xs text-ink-dim ml-2">
                ({cache.alternative_engine})
              </span>
            )}
          </Field>
        )}
      </div>
      <AssumptionsList items={cache.assumptions} />
    </section>
  );
}

function LoadBalancerSection({
  lb,
}: {
  lb: SystemDesignRecommendation["load_balancer"];
}) {
  return (
    <section>
      <SectionHeader label="load balancer" />
      {lb.needed ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {lb.load_balancer_type && (
            <Field label="type">
              <MonoValue value={lb.load_balancer_type} />
            </Field>
          )}
          <Field label="rationale">
            <p className="font-sans text-sm text-ink-muted leading-relaxed">
              {lb.why}
            </p>
          </Field>
        </div>
      ) : (
        <p className="font-sans text-sm text-ink-muted">{lb.why}</p>
      )}
    </section>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────

export function ResultReport({ sdr }: Props) {
  return (
    <div className="animate-fade-in space-y-0">
      {/* ── Architecture summary — most prominent ── */}
      <section className="py-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="font-mono text-xs text-amber uppercase tracking-widest">
            architecture summary
          </span>
          <span className="flex-1 border-t border-amber/20" />
        </div>
        <ArchSummary text={sdr.architecture_summary} />
      </section>

      <hr className="console-rule" />

      {/* ── Tier sections ── */}
      <div className="py-6">
        <ComputeSection compute={sdr.compute} />
      </div>

      <hr className="console-rule" />

      <div className="py-6">
        <DatabaseSection database={sdr.database} />
      </div>

      <hr className="console-rule" />

      <div className="py-6">
        <CacheSection cache={sdr.cache} />
      </div>

      <hr className="console-rule" />

      <div className="py-6">
        <LoadBalancerSection lb={sdr.load_balancer} />
      </div>
    </div>
  );
}
