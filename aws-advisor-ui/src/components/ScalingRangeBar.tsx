"use client";

interface Props {
  minInstances: number;
  maxInstances: number;
  scalingRecommendation: string;
}

const C = {
  track: "rgba(237,237,234,0.07)",
  fill: "#E8A33D",
  label: "rgba(237,237,234,0.30)",
  valuePrimary: "#EDEDEA",
  font: "'IBM Plex Mono', ui-monospace, monospace",
} as const;

const SVG_W = 320;
const SVG_H = 40;
const BAR_H = 10;
const BAR_Y = 10;
const LABEL_Y = BAR_Y + BAR_H + 16;

export function ScalingRangeBar({ minInstances, maxInstances, scalingRecommendation }: Props) {
  // Render a proportional bar. Scale: 0 to max+1 so the bar never fills 100%.
  const scale = maxInstances + 1;
  const fullW = SVG_W - 2; // 1px inset each side
  const minX = (minInstances / scale) * fullW;
  const maxX = (maxInstances / scale) * fullW;
  const barW = Math.max(maxX - minX, 2);

  const singleInstance = minInstances === maxInstances;

  return (
    <div className="w-full">
      <svg
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        aria-label={`Scaling range: ${minInstances} to ${maxInstances} instances`}
        style={{ display: "block", maxWidth: "100%" }}
      >
        {/* Track */}
        <rect
          x={1} y={BAR_Y}
          width={fullW} height={BAR_H}
          fill={C.track}
          rx={1}
        />

        {/* Filled range */}
        <rect
          x={1 + minX} y={BAR_Y}
          width={barW} height={BAR_H}
          fill={C.fill}
          rx={1}
        />

        {/* Min label */}
        <text
          x={1 + minX + (singleInstance ? barW / 2 : 0)}
          y={LABEL_Y}
          textAnchor={singleInstance ? "middle" : "start"}
          fill={C.valuePrimary}
          fontSize={10}
          fontFamily={C.font}
        >
          {minInstances}
        </text>

        {/* Max label — only if different from min */}
        {!singleInstance && (
          <text
            x={1 + maxX}
            y={LABEL_Y}
            textAnchor="end"
            fill={C.valuePrimary}
            fontSize={10}
            fontFamily={C.font}
          >
            {maxInstances}
          </text>
        )}

        {/* "instances" suffix */}
        <text
          x={SVG_W}
          y={LABEL_Y}
          textAnchor="end"
          fill={C.label}
          fontSize={9}
          fontFamily={C.font}
          letterSpacing="0.04em"
        >
          instances
        </text>
      </svg>

      {/* Scaling recommendation note */}
      <p
        className="font-sans text-xs leading-relaxed mt-1"
        style={{ color: "rgba(237,237,234,0.35)" }}
      >
        {scalingRecommendation}
      </p>
    </div>
  );
}
