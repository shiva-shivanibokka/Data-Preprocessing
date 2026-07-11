"use client";

import { useEffect, useState } from "react";

/** Fetch a static JSON artifact from /public once. */
export function useJson<T>(path: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(path)
      .then((r) => r.json())
      .then((d) => alive && setData(d))
      .catch(() => alive && setData(null));
    return () => {
      alive = false;
    };
  }, [path]);
  return data;
}

const COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#a3e635", "#fbbf24", "#fb7185"];
export const color = (i: number) => COLORS[i % COLORS.length];

/** Vertical bar chart with value labels. */
export function Bars({
  items,
  height = 260,
  fmt = (v: number) => `${v}`,
}: {
  items: { label: string; value: number; color?: string }[];
  height?: number;
  fmt?: (v: number) => string;
}) {
  const W = 520;
  const pad = 30;
  const bottom = 46;
  const max = Math.max(...items.map((d) => d.value), 1);
  const bw = (W - 2 * pad) / items.length;
  const sy = (v: number) => height - bottom - (v / max) * (height - bottom - 14);
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${height}`}>
        {items.map((d, i) => {
          const x = pad + i * bw;
          const y = sy(d.value);
          return (
            <g key={i}>
              <rect
                x={x + bw * 0.12}
                y={y}
                width={bw * 0.76}
                height={height - bottom - y}
                rx={4}
                fill={d.color ?? color(i)}
                opacity={0.9}
              />
              <text x={x + bw / 2} y={y - 5} fill="var(--text)" fontSize={11} textAnchor="middle">
                {fmt(d.value)}
              </text>
              <text
                x={x + bw / 2}
                y={height - bottom + 16}
                fill="var(--muted)"
                fontSize={10}
                textAnchor="middle"
              >
                {d.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Scatter plot with two-class coloring. */
export function Scatter({
  points,
  xlabel,
  ylabel,
  height = 340,
}: {
  points: { x: number; y: number; c: string }[];
  xlabel: string;
  ylabel: string;
  height?: number;
}) {
  const W = 520;
  const pad = 46;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs),
    ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const sx = (v: number) => pad + ((v - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (v: number) => height - pad - ((v - ymin) / (ymax - ymin || 1)) * (height - 2 * pad);
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${height}`}>
        <line x1={pad} y1={height - pad} x2={W - pad} y2={height - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="var(--border)" />
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={p.c} opacity={0.72} />
        ))}
        <text x={W / 2} y={height - 8} fill="var(--muted)" fontSize={11} textAnchor="middle">
          {xlabel}
        </text>
        <text
          x={14}
          y={height / 2}
          fill="var(--muted)"
          fontSize={11}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
        >
          {ylabel}
        </text>
      </svg>
    </div>
  );
}

/** A single self-scaled vertical boxplot (min/q1/median/q3/max). */
export function MiniBox({
  label,
  stats,
  tone = "#a78bfa",
}: {
  label: string;
  stats: { min: number; q1: number; median: number; q3: number; max: number };
  tone?: string;
}) {
  const W = 120;
  const H = 220;
  const pad = 24;
  const lo = stats.min;
  const hi = stats.max;
  const s = (v: number) => H - pad - ((v - lo) / (hi - lo || 1)) * (H - 2 * pad);
  const cx = W / 2;
  return (
    <div className="chart-box" style={{ padding: ".7rem" }}>
      <p className="section-label" style={{ marginBottom: ".3rem", justifyContent: "center" }}>
        {label}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`}>
        <line x1={cx} y1={s(stats.max)} x2={cx} y2={s(stats.min)} stroke="var(--muted)" strokeWidth={1.5} />
        <rect x={cx - 24} y={s(stats.q3)} width={48} height={Math.max(s(stats.q1) - s(stats.q3), 1)} rx={4} fill={tone} opacity={0.35} stroke={tone} />
        <line x1={cx - 24} y1={s(stats.median)} x2={cx + 24} y2={s(stats.median)} stroke={tone} strokeWidth={2.5} />
        <line x1={cx - 12} y1={s(stats.max)} x2={cx + 12} y2={s(stats.max)} stroke="var(--muted)" />
        <line x1={cx - 12} y1={s(stats.min)} x2={cx + 12} y2={s(stats.min)} stroke="var(--muted)" />
      </svg>
      <p className="note" style={{ textAlign: "center", fontSize: ".72rem", marginTop: ".3rem" }}>
        {stats.min.toFixed(2)} … {stats.max.toFixed(2)}
      </p>
    </div>
  );
}
