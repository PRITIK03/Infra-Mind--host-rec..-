"use client";

import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

// A minimal dark theme matching the console palette — no colors except
// amber for keywords, off-white for text, muted for comments.
const consoleTheme: { [key: string]: React.CSSProperties } = {
  'code[class*="language-"]': {
    color: "#ededea",
    fontFamily: "var(--font-ibm-plex-mono), ui-monospace, monospace",
    fontSize: "0.8125rem",
    lineHeight: "1.6",
    background: "transparent",
  },
  'pre[class*="language-"]': {
    background: "transparent",
    margin: 0,
    padding: 0,
    overflow: "auto",
  },
  comment: { color: "rgba(237,237,234,0.35)", fontStyle: "italic" },
  prolog: { color: "rgba(237,237,234,0.35)" },
  doctype: { color: "rgba(237,237,234,0.35)" },
  cdata: { color: "rgba(237,237,234,0.35)" },
  punctuation: { color: "rgba(237,237,234,0.55)" },
  property: { color: "#e8a33d" },
  tag: { color: "#e8a33d" },
  boolean: { color: "#e8a33d" },
  number: { color: "#e8a33d" },
  constant: { color: "#e8a33d" },
  symbol: { color: "#e8a33d" },
  deleted: { color: "#f87171" },
  selector: { color: "#ededea" },
  "attr-name": { color: "rgba(237,237,234,0.75)" },
  string: { color: "rgba(237,237,234,0.85)" },
  char: { color: "rgba(237,237,234,0.85)" },
  builtin: { color: "rgba(237,237,234,0.75)" },
  inserted: { color: "#86efac" },
  operator: { color: "rgba(237,237,234,0.55)" },
  entity: { color: "#e8a33d" },
  url: { color: "rgba(237,237,234,0.75)" },
  variable: { color: "#ededea" },
  atrule: { color: "#e8a33d" },
  "attr-value": { color: "rgba(237,237,234,0.85)" },
  function: { color: "#ededea" },
  keyword: { color: "#e8a33d" },
  regex: { color: "rgba(237,237,234,0.75)" },
  important: { color: "#e8a33d", fontWeight: "bold" },
  bold: { fontWeight: "bold" },
  italic: { fontStyle: "italic" },
};

interface Props {
  files: Record<string, string>;
}

export function TerraformViewer({ files }: Props) {
  const fileNames = Object.keys(files);
  const [activeFile, setActiveFile] = useState<string>(fileNames[0] ?? "");
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const activeContent = files[activeFile] ?? "";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore — clipboard access can be denied
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const JSZip = (await import("jszip")).default;
      const zip = new JSZip();
      const folder = zip.folder("terraform");
      for (const [name, content] of Object.entries(files)) {
        folder?.file(name, content);
      }
      const blob = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "terraform.zip";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("zip failed", err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Section header */}
      <div className="flex items-center gap-2 mb-4">
        <span className="font-mono text-xs text-ink-muted uppercase tracking-widest">
          terraform output
        </span>
        <span className="flex-1 border-t border-border-subtle" />
        <button
          onClick={handleDownload}
          disabled={downloading}
          className={[
            "flex items-center gap-1.5 border border-border-subtle px-3 py-1",
            "font-mono text-xs text-ink-muted uppercase tracking-wider",
            "transition-colors duration-150",
            "hover:text-amber hover:border-amber/40",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          ].join(" ")}
        >
          {downloading ? "packing…" : "↓ download .zip"}
        </button>
      </div>

      {/* File tabs */}
      <div className="flex border-b border-border-subtle">
        {fileNames.map((name) => (
          <button
            key={name}
            onClick={() => setActiveFile(name)}
            className={[
              "px-4 py-2 font-mono text-xs transition-colors duration-150 border-r border-border-subtle",
              activeFile === name
                ? "text-amber border-b border-b-amber -mb-px bg-amber/5"
                : "text-ink-dim hover:text-ink-muted",
            ].join(" ")}
          >
            {name}
          </button>
        ))}
        {/* Copy button — right-aligned */}
        <div className="ml-auto flex items-center pr-1">
          <button
            onClick={handleCopy}
            className={[
              "px-3 py-1.5 font-mono text-xs transition-colors duration-150",
              copied
                ? "text-amber"
                : "text-ink-dim hover:text-ink-muted",
            ].join(" ")}
            title="Copy to clipboard"
          >
            {copied ? "copied ✓" : "copy"}
          </button>
        </div>
      </div>

      {/* Code viewer */}
      <div className="border border-t-0 border-border-subtle overflow-auto max-h-[520px] bg-white/[0.02]">
        <div className="p-4 min-w-0">
          <SyntaxHighlighter
            language="hcl"
            style={consoleTheme}
            showLineNumbers
            lineNumberStyle={{
              color: "rgba(237,237,234,0.18)",
              fontSize: "0.7rem",
              minWidth: "2.5em",
              paddingRight: "1em",
              userSelect: "none",
            }}
            wrapLongLines={false}
            customStyle={{
              background: "transparent",
              padding: 0,
              margin: 0,
              fontSize: "0.8125rem",
            }}
          >
            {activeContent}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
}
