interface Props {
  level: "low" | "medium" | "high";
}

const MAP = {
  high:   { dot: "bg-amber",               label: "high confidence" },
  medium: { dot: "bg-ink-muted",           label: "medium confidence" },
  low:    { dot: "bg-red-400/60",          label: "low confidence" },
} as const;

export function ConfidenceDot({ level }: Props) {
  const { dot, label } = MAP[level] ?? MAP.medium;
  return (
    <span
      className="inline-flex items-center gap-1.5"
      title={label}
      aria-label={label}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot}`} />
      <span className="font-mono text-xs text-ink-dim">{level}</span>
    </span>
  );
}
