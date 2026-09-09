"use client";

import type { SystemDesignRecommendation } from "@/lib/types";

interface Props {
  sdr: SystemDesignRecommendation;
}

const C = {
  segFilled: "#E8A33D",
  segEmpty: "rgba(237,237,234,0.08)",
  label: "rgba(237,237,234,0.35)",
  font: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

// Opacity per segment when filled — low/med/high use increasing opacity
// on the single amber accent, never a different hue.
const FILLED_OPACITY = [0.40, 0.70, 1.0];

const SEGMENT_W = 14;
const SEGMENT_H = 6;
const SEGMENT_GAP = 3;
const ROW_GAP = 10;
const LABEL_W = 90;
const SVG_PAD = 4;

type Level = "low" | "medium" | "high";

function levelToIndex(level: Level): number {
  return level === "low" ? 0 : level === "medium" ? 1 : 2;
}

interface TierRow {
  label: string;
  level: Level;
}

export function ConfidenceStrip({ sdr }: Props) {
  const rows: TierRow[] = [];

  // Always show compute
  rows.push({ label: "compute", level: sdr.compute.confidence as Level });

  if (sdr.database.needed) {
    rows.push({ label: "database", level: sdr.database.confidence as Level });
  }
  if (sdr.cache.needed) {
    rows.push({ label: "cache", level: sdr.cache.confidence as Level });
  }
  // Load balancer has no confidence field — skip

  const stripW = LABEL_W + (SEGMENT_W + SEGMENT_GAP) * 3;
  const svgH = SVG_PAD * 2 + rows.length * SEGMENT_H + (rows.length - 1) * ROW_GAP;

  return (
    <svg
      width={stripW}
      height={svgH}
      viewBox={`0 0 ${stripW} ${svgH}`}
      aria-label="Confidence levels by tier"
      style={{ display: "block" }}
    >
      {rows.map((row, ri) => {
        const filledUpTo = levelToIndex(row.level);
        const rowY = SVG_PAD + ri * (SEGMENT_H + ROW_GAP);

        return (
          <g key={row.label}>
            {/* Tier label */}
            <text
              x={0}
              y={rowY + SEGMENT_H - 1}
              fill={C.label}
              fontSize={9}
              fontFamily={C.font}
              letterSpacing="0.06em"
            >
              {row.label.toUpperCase()}
            </text>

            {/* 3 segments */}
            {[0, 1, 2].map((si) => {
              const filled = si <= filledUpTo;
              return (
                <rect
                  key={si}
                  x={LABEL_W + si * (SEGMENT_W + SEGMENT_GAP)}
                  y={rowY}
                  width={SEGMENT_W}
                  height={SEGMENT_H}
                  rx={1}
                  fill={filled ? C.segFilled : C.segEmpty}
                  fillOpacity={filled ? FILLED_OPACITY[si] : 1}
                />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
