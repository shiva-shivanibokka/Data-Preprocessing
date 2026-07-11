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

/** Small "how to read this chart" box shown under a graph. */
export function HowToRead({ children }: { children: React.ReactNode }) {
  return (
    <p className="howto">
      <span className="howto-tag">How to read</span>
      {children}
    </p>
  );
}

/** Vertical bar chart with value labels and per-bar hover tooltips. */
export function Bars({
  items,
  height = 260,
  fmt = (v: number) => `${v}`,
}: {
  items: { label: string; value: number; color?: string; title?: string }[];
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
              >
                <title>{d.title ?? `${d.label}: ${fmt(d.value)}`}</title>
              </rect>
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

/** Histogram from raw values with an optional cutoff line; bars beyond the cutoff turn red. */
export function Histogram({
  values,
  bins = 24,
  height = 240,
  cutoff,
  cutoffLabel,
  unit = "",
}: {
  values: number[];
  bins?: number;
  height?: number;
  cutoff?: number;
  cutoffLabel?: string;
  unit?: string;
}) {
  const W = 520;
  const pad = 34;
  const bottom = 40;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const step = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => {
    const b = Math.min(bins - 1, Math.floor((v - lo) / step));
    counts[b]++;
  });
  const maxC = Math.max(...counts, 1);
  const bw = (W - 2 * pad) / bins;
  const sx = (v: number) => pad + ((v - lo) / (hi - lo || 1)) * (W - 2 * pad);
  const sy = (c: number) => height - bottom - (c / maxC) * (height - bottom - 14);
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${height}`}>
        {counts.map((c, i) => {
          const binLo = lo + i * step;
          const binHi = binLo + step;
          const beyond = cutoff !== undefined && binLo >= cutoff;
          return (
            <rect
              key={i}
              x={pad + i * bw + 1}
              y={sy(c)}
              width={bw - 2}
              height={height - bottom - sy(c)}
              rx={2}
              fill={beyond ? "#fb7185" : "#4C78A8"}
              opacity={0.9}
            >
              <title>{`${binLo.toFixed(0)}–${binHi.toFixed(0)}${unit}: ${c}`}</title>
            </rect>
          );
        })}
        {cutoff !== undefined && cutoff <= hi && (
          <g>
            <line x1={sx(cutoff)} y1={10} x2={sx(cutoff)} y2={height - bottom} stroke="#fb7185" strokeWidth={2} strokeDasharray="5 4" />
            {cutoffLabel && (
              <text x={sx(cutoff) - 6} y={20} fill="#fb7185" fontSize={10} textAnchor="end">
                {cutoffLabel}
              </text>
            )}
          </g>
        )}
        <text x={W / 2} y={height - 6} fill="var(--muted)" fontSize={11} textAnchor="middle">
          value{unit ? ` (${unit})` : ""} →
        </text>
      </svg>
    </div>
  );
}

/** 1D jittered strip plot with a shaded "normal" band [low, high]; points outside turn red. */
export function Strip({
  values,
  low,
  high,
  height = 150,
  unit = "",
}: {
  values: number[];
  low: number;
  high: number;
  height: number;
  unit?: string;
}) {
  const W = 520;
  const pad = 34;
  const lo = Math.min(...values, low);
  const hi = Math.max(...values, high);
  const sx = (v: number) => pad + ((v - lo) / (hi - lo || 1)) * (W - 2 * pad);
  const top = 22;
  const band = height - 44;
  // deterministic pseudo-jitter so points don't jump between renders
  const jit = (i: number) => top + ((i * 73 + 17) % 100) / 100 * band;
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${W} ${height}`}>
        <rect x={sx(low)} y={top} width={Math.max(sx(high) - sx(low), 1)} height={band} fill="#4C78A8" opacity={0.12} rx={6} />
        <line x1={sx(low)} y1={top} x2={sx(low)} y2={top + band} stroke="#4C78A8" strokeDasharray="4 4" opacity={0.6} />
        <line x1={sx(high)} y1={top} x2={sx(high)} y2={top + band} stroke="#4C78A8" strokeDasharray="4 4" opacity={0.6} />
        {values.map((v, i) => {
          const out = v < low || v > high;
          return (
            <circle key={i} cx={sx(v)} cy={jit(i)} r={3.5} fill={out ? "#fb7185" : "#4C78A8"} opacity={out ? 0.9 : 0.5}>
              <title>{`${v.toFixed(2)}${unit}${out ? " — outlier" : ""}`}</title>
            </circle>
          );
        })}
        <text x={sx(high) + 6} y={top + 12} fill="#4C78A8" fontSize={10}>normal range</text>
        <text x={W / 2} y={height - 6} fill="var(--muted)" fontSize={11} textAnchor="middle">
          value{unit ? ` (${unit})` : ""} →
        </text>
      </svg>
    </div>
  );
}

/** Scatter plot with two-class coloring and per-point hover tooltips. */
export function Scatter({
  points,
  xlabel,
  ylabel,
  height = 340,
}: {
  points: { x: number; y: number; c: string; title?: string }[];
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
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={p.c} opacity={0.72}>
            <title>{p.title ?? `${xlabel} ${p.x}, ${ylabel} ${p.y}`}</title>
          </circle>
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

/** A single self-scaled vertical boxplot (min/q1/median/q3/max) with a hover summary. */
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
  const summary = `${label}
max ${stats.max.toFixed(2)}
Q3  ${stats.q3.toFixed(2)}
med ${stats.median.toFixed(2)}
Q1  ${stats.q1.toFixed(2)}
min ${stats.min.toFixed(2)}`;
  return (
    <div className="chart-box" style={{ padding: ".7rem" }}>
      <p className="section-label" style={{ marginBottom: ".3rem", justifyContent: "center" }}>
        {label}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`}>
        <title>{summary}</title>
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
