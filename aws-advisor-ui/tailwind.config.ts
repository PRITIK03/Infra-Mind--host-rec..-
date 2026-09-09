import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Engineering console palette — single source of truth
        canvas: "#0F0F0E",      // near-black background
        ink: "#EDEDEA",         // off-white text
        amber: "#E8A33D",       // muted amber accent — used sparingly
        "amber-dim": "#B5792A", // darker amber for hover states
        "border-subtle": "rgba(237,237,234,0.10)", // thin rule color
        "border-faint": "rgba(237,237,234,0.05)",
        "ink-muted": "rgba(237,237,234,0.45)", // secondary text
        "ink-dim": "rgba(237,237,234,0.25)",   // tertiary / disabled
        "stage-done": "rgba(232,163,61,0.18)", // amber tint for completed stages
      },
      fontFamily: {
        // IBM Plex Sans — prose, labels, UI copy
        sans: ["var(--font-ibm-plex-sans)", "system-ui", "sans-serif"],
        // IBM Plex Mono — instance types, resource names, code, Terraform
        mono: ["var(--font-ibm-plex-mono)", "ui-monospace", "monospace"],
      },
      borderColor: {
        DEFAULT: "rgba(237,237,234,0.10)",
      },
      animation: {
        "cursor-blink": "cursor-blink 1.1s step-end infinite",
        "fade-in": "fade-in 0.3s ease forwards",
        "slide-in": "slide-in 0.25s ease forwards",
      },
      keyframes: {
        "cursor-blink": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
