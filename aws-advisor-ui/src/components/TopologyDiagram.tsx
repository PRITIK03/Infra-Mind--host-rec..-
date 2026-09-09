"use client";

import type { SystemDesignRecommendation } from "@/lib/types";

interface Props {
  sdr: SystemDesignRecommendation;
}

// Palette constants — single source of truth matching globals.css
const C = {
  bg: "#0F0F0E",
  box: "rgba(237,237,234,0.06)",
  boxStroke: "rgba(237,237,234,0.18)",
  line: "rgba(237,237,234,0.18)",
  labelMuted: "rgba(237,237,234,0.35)",
  labelPrimary: "#EDEDEA",
  amber: "#E8A33D",
  font: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

const BOX_W = 130;
const BOX_H = 52;
const GAP_X = 36; // horizontal gap between boxes
const PAD_Y = 20; // top/bottom padding inside SVG
const PAD_X = 16;

interface Node {
  id: string;
  label: string;   // small caps category label
  value: string;   // monospace instance type / lb type
}

export function TopologyDiagram({ sdr }: Props) {
  // Build node list conditionally from real recommendation data
  const nodes: Node[] = [];

  nodes.push({ id: "users", label: "users", value: "internet" });

  if (sdr.load_balancer.needed) {
    nodes.push({
      id: "lb",
      label: "load balancer",
      value: sdr.load_balancer.load_balancer_type ?? "ALB",
    });
  }

  nodes.push({
    id: "compute",
    label: "compute (ASG)",
    value: sdr.compute.recommended_instance,
  });

  if (sdr.cache.needed && sdr.cache.recommended_instance) {
    nodes.push({
      id: "cache",
      label: `cache · ${sdr.cache.engine ?? ""}`,
      value: sdr.cache.recommended_instance,
    });
  }

  if (sdr.database.needed && sdr.database.recommended_instance) {
    nodes.push({
      id: "db",
      label: `database · ${sdr.database.engine_suggestion ?? ""}`,
      value: sdr.database.recommended_instance,
    });
  }

  const totalW = nodes.length * BOX_W + (nodes.length - 1) * GAP_X + PAD_X * 2;
  const totalH = BOX_H + PAD_Y * 2;

  return (
    <div className="w-full overflow-x-auto">
      <svg
        width={totalW}
        height={totalH}
        viewBox={`0 0 ${totalW} ${totalH}`}
        aria-label="Architecture topology diagram"
        style={{ display: "block", maxWidth: "100%" }}
      >
        {/* Connector lines first (behind boxes) */}
        {nodes.slice(0, -1).map((_, i) => {
          const x1 = PAD_X + i * (BOX_W + GAP_X) + BOX_W;
          const x2 = PAD_X + (i + 1) * (BOX_W + GAP_X);
          const cy = PAD_Y + BOX_H / 2;
          const mx = (x1 + x2) / 2;
          return (
            <g key={`line-${i}`}>
              {/* Line */}
              <line
                x1={x1} y1={cy}
                x2={x2} y2={cy}
                stroke={C.line}
                strokeWidth={1}
              />
              {/* Arrowhead */}
              <polygon
                points={`${x2},${cy} ${x2 - 6},${cy - 3} ${x2 - 6},${cy + 3}`}
                fill={C.line}
              />
            </g>
          );
        })}

        {/* Boxes */}
        {nodes.map((node, i) => {
          const x = PAD_X + i * (BOX_W + GAP_X);
          const y = PAD_Y;
          const isCompute = node.id === "compute";
          const strokeColor = isCompute ? C.amber : C.boxStroke;
          const strokeW = isCompute ? 1.5 : 1;

          return (
            <g key={node.id}>
              <rect
                x={x} y={y}
                width={BOX_W} height={BOX_H}
                fill={C.box}
                stroke={strokeColor}
                strokeWidth={strokeW}
                rx={2}
              />
              {/* Category label */}
              <text
                x={x + BOX_W / 2}
                y={y + 14}
                textAnchor="middle"
                fill={C.labelMuted}
                fontSize={8}
                fontFamily={C.font}
                letterSpacing="0.08em"
                style={{ textTransform: "uppercase" }}
              >
                {node.label}
              </text>
              {/* Instance/type value — truncated if too long */}
              <text
                x={x + BOX_W / 2}
                y={y + 33}
                textAnchor="middle"
                fill={isCompute ? C.amber : C.labelPrimary}
                fontSize={11}
                fontFamily={C.font}
                letterSpacing="-0.01em"
              >
                {node.value.length > 16
                  ? node.value.slice(0, 14) + "…"
                  : node.value}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
