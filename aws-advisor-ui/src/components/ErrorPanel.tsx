"use client";

interface Props {
  message: string;
  onRetry: () => void;
}

export function ErrorPanel({ message, onRetry }: Props) {
  return (
    <div className="animate-slide-in">
      <div className="flex items-center gap-2 mb-4">
        <span className="font-mono text-xs text-red-400/70 uppercase tracking-widest">
          error
        </span>
        <span className="flex-1 border-t border-red-400/15" />
      </div>

      <div className="border border-red-400/20 p-4 mb-5">
        <p className="font-mono text-sm text-red-300/80 leading-relaxed whitespace-pre-wrap break-words">
          {message}
        </p>
      </div>

      <button
        onClick={onRetry}
        className={[
          "border border-border-subtle px-4 py-2",
          "font-mono text-xs text-ink-muted uppercase tracking-wider",
          "transition-colors duration-150",
          "hover:text-amber hover:border-amber/40",
        ].join(" ")}
      >
        ↩ start over
      </button>
    </div>
  );
}
