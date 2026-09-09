"use client";

import { useEffect, useState } from "react";
import { getStats } from "@/lib/api";

interface Stats {
  ec2: number;
  rds: number;
  cache: number;
}

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

export function LiveStatsReadout() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getStats()
      .then((s) => { if (!cancelled) { setStats(s); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <span className="font-mono text-xs text-ink-dim animate-pulse">
        loading instance data…
      </span>
    );
  }

  if (!stats) return null;

  return (
    <span className="font-mono text-xs text-ink-dim">
      <span className="text-ink-muted">{fmt(stats.ec2)}</span> EC2
      {" · "}
      <span className="text-ink-muted">{fmt(stats.rds)}</span> RDS
      {" · "}
      <span className="text-ink-muted">{fmt(stats.cache)}</span> cache types tracked live
    </span>
  );
}
