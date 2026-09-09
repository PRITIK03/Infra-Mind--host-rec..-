"use client";

import { useState } from "react";
import type { InstanceCandidate } from "@/lib/types";

interface Props {
  candidates: InstanceCandidate[];
  recommendedInstance: string;
  alternativeInstance?: string | null;
}

const C = {
  amber: "#E8A33D",
  amberDim: "rgba(232,163,61,0.45)",
  amberFaint: "rgba(232,163,61,0.15)",
  axisLine: "rgba(237,237,234,0.12)",
  axisLabel: "rgba(237,237,234,0.30)",
  dotDefault: "rgba(237,237,234,0.15)",
  tooltip: "#1a1a19",
  tooltipBorder: "rgba(237,237,234,0.18)",
  tooltipText: "#EDEDEA",
  font: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

const SVG_W = 480;
const SVG_H = 280;
const MARGIN = { top: 16, right: 16, bottom: 44, left: 52 };

const plotW = SVG_W - MARGIN.left - MARGIN.right;
const plotH = SVG_H - MARGIN.top - MARGIN.bottom;

function linear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function niceMax(raw: number): number {
  if (raw <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return Math.ceil(raw / mag) * mag;
}

function axisTicks(max: number, count = 5): number[] {
  const step = max / count;
  return Array.from({ length: count + 1 }, (_, i) => Math.round(i * step));
}

export function CandidateLandscape({
  candidates,
  recommendedInstance,
  alternativeInstance,
}: Props) {
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    label: string;
  } | null>(null);

  if (!candidates.length) return null;

  const maxVcpu = niceMax(Math.max(...candidates.map((c) => c.vcpu)));
  const maxMem = niceMax(Math.max(...candidates.map((c) => c.memory_gib)));

  const scaleX = linear([0, maxVcpu], [0, plotW]);
  const scaleY = linear([0, maxMem], [plotH, 0]); // inverted: 0 at bottom

  const xTicks = axisTicks(maxVcpu, 5);
  const yTicks = axisTicks(maxMem, 5);

  return (
    <div className="w-full overflow-x-auto">
      <svg
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        aria-label="Candidate instance scatter plot — vCPU vs memory"
        style={{ display: "block", maxWidth: "100%", userSelect: "none" }}
        onMouseLeave={() => setTooltip(null)}
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* X axis */}
          <line
            x1={0} y1={plotH}
            x2={plotW} y2={plotH}
            stroke={C.axisLine} strokeWidth={1}
          />
          {xTicks.map((v) => (
            <g key={`x${v}`} transform={`translate(${scaleX(v)},${plotH})`}>
              <line y1={0} y2={5} stroke={C.axisLine} strokeWidth={1} />
              <text
                y={18}
                textAnchor="middle"
                fill={C.axisLabel}
                fontSize={9}
                fontFamily={C.font}
              >
                {v}
              </text>
            </g>
          ))}
          {/* X axis label */}
          <text
            x={plotW / 2}
            y={plotH + 36}
            textAnchor="middle"
            fill={C.axisLabel}
            fontSize={9}
            fontFamily={C.font}
            letterSpacing="0.06em"
          >
            vCPU
          </text>

          {/* Y axis */}
          <line
            x1={0} y1={0}
            x2={0} y2={plotH}
            stroke={C.axisLine} strokeWidth={1}
          />
          {yTicks.map((v) => (
            <g key={`y${v}`} transform={`translate(0,${scaleY(v)})`}>
              <line x1={-5} x2={0} stroke={C.axisLine} strokeWidth={1} />
              <text
                x={-9}
                textAnchor="end"
                dominantBaseline="middle"
                fill={C.axisLabel}
                fontSize={9}
                fontFamily={C.font}
              >
                {v}
              </text>
            </g>
          ))}
          {/* Y axis label */}
          <text
            transform={`translate(-38,${plotH / 2}) rotate(-90)`}
            textAnchor="middle"
            fill={C.axisLabel}
            fontSize={9}
            fontFamily={C.font}
            letterSpacing="0.06em"
          >
            Memory GiB
          </text>

          {/* Dots — render in layers: background, alt, recommended on top */}
          {candidates
            .filter(
              (c) =>
                c.instance_type !== recommendedInstance &&
                c.instance_type !== alternativeInstance
            )
            .map((c) => (
              <circle
                key={c.instance_type}
                cx={scaleX(c.vcpu)}
                cy={scaleY(c.memory_gib)}
                r={3}
                fill={C.dotDefault}
                style={{ cursor: "pointer" }}
                onMouseEnter={(e) => {
                  const svgRect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                  setTooltip({
                    x: scaleX(c.vcpu),
                    y: scaleY(c.memory_gib) - 10,
                    label: c.instance_type,
                  });
                }}
              />
            ))}

          {/* Alternative dot */}
          {alternativeInstance &&
            candidates
              .filter((c) => c.instance_type === alternativeInstance)
              .map((c) => (
                <circle
                  key={c.instance_type}
                  cx={scaleX(c.vcpu)}
                  cy={scaleY(c.memory_gib)}
                  r={5}
                  fill={C.amberDim}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() =>
                    setTooltip({
                      x: scaleX(c.vcpu),
                      y: scaleY(c.memory_gib) - 12,
                      label: c.instance_type,
                    })
                  }
                />
              ))}

          {/* Recommended dot — largest, amber */}
          {candidates
            .filter((c) => c.instance_type === recommendedInstance)
            .map((c) => (
              <g key={c.instance_type}>
                {/* Outer ring for emphasis */}
                <circle
                  cx={scaleX(c.vcpu)}
                  cy={scaleY(c.memory_gib)}
                  r={9}
                  fill="none"
                  stroke={C.amberFaint}
                  strokeWidth={1}
                />
                <circle
                  cx={scaleX(c.vcpu)}
                  cy={scaleY(c.memory_gib)}
                  r={6}
                  fill={C.amber}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() =>
                    setTooltip({
                      x: scaleX(c.vcpu),
                      y: scaleY(c.memory_gib) - 14,
                      label: c.instance_type,
                    })
                  }
                />
              </g>
            ))}

          {/* Tooltip */}
          {tooltip && (
            <g transform={`translate(${tooltip.x},${tooltip.y})`} style={{ pointerEvents: "none" }}>
              {/* Measure approximate text width */}
              <rect
                x={-tooltip.label.length * 3.2 - 6}
                y={-20}
                width={tooltip.label.length * 6.4 + 12}
                height={18}
                rx={2}
                fill={C.tooltip}
                stroke={C.tooltipBorder}
                strokeWidth={1}
              />
              <text
                x={0}
                y={-7}
                textAnchor="middle"
                fill={C.tooltipText}
                fontSize={10}
                fontFamily={C.font}
              >
                {tooltip.label}
              </text>
            </g>
          )}
        </g>

        {/* Legend */}
        <g transform={`translate(${MARGIN.left + plotW - 110}, ${MARGIN.top + 4})`}>
          <circle cx={5} cy={5} r={5} fill={C.amber} />
          <text x={14} y={9} fill={C.axisLabel} fontSize={9} fontFamily={C.font}>recommended</text>
          {alternativeInstance && (
            <>
              <circle cx={5} cy={22} r={4} fill={C.amberDim} />
              <text x={14} y={26} fill={C.axisLabel} fontSize={9} fontFamily={C.font}>alternative</text>
            </>
          )}
          <circle cx={5} cy={alternativeInstance ? 39 : 22} r={3} fill={C.dotDefault} />
          <text
            x={14}
            y={alternativeInstance ? 43 : 26}
            fill={C.axisLabel}
            fontSize={9}
            fontFamily={C.font}
          >
            candidates
          </text>
        </g>
      </svg>
    </div>
  );
}
