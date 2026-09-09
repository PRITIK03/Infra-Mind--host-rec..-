"use client";

import { useRef, useState } from "react";

interface Props {
  question: string;
  onAnswer: (answer: string) => void;
  disabled?: boolean;
}

export function AwaitingInputPanel({ question, onAnswer, disabled }: Props) {
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onAnswer(trimmed);
    setValue("");
  };

  return (
    <div className="animate-slide-in">
      {/* Question */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="font-mono text-xs text-ink-muted uppercase tracking-widest">
            follow-up
          </span>
          <span className="flex-1 border-t border-border-subtle" />
        </div>
        <p className="text-ink text-sm leading-relaxed">{question}</p>
      </div>

      {/* Answer input */}
      <form onSubmit={handleSubmit} className="flex gap-0">
        <div className="flex items-center border border-border-subtle bg-transparent px-3 text-amber font-mono text-sm shrink-0">
          &gt;
        </div>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="type your answer…"
          disabled={disabled}
          autoFocus
          className={[
            "flex-1 bg-transparent border-t border-b border-border-subtle",
            "font-mono text-sm text-ink placeholder:text-ink-dim",
            "px-3 py-2.5 outline-none",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          ].join(" ")}
        />
        <button
          type="submit"
          disabled={!value.trim() || disabled}
          className={[
            "border border-border-subtle border-l-0 px-4 py-2.5",
            "font-mono text-xs text-ink-muted uppercase tracking-wider",
            "transition-colors duration-150",
            "hover:text-amber hover:border-amber/40",
            "disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:text-ink-muted disabled:hover:border-border-subtle",
          ].join(" ")}
        >
          send
        </button>
      </form>
    </div>
  );
}
