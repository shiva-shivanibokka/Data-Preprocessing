"use client";

import { useEffect, useRef, useState } from "react";

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

/* ---------- custom hover tooltip (bright, smooth, follows the cursor) ---------- */
type TipState = { x: number; y: number; text: string } | null;

export function useHoverTip() {
  const [tip, setTip] = useState<TipState>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const move = (e: React.MouseEvent, text: string) => {
    const r = wrap.current?.getBoundingClientRect();
    if (!r) return;
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, text });
  };
  const leave = () => setTip(null);
  return { tip, wrap, move, leave };
}

export function TipBox({ tip }: { tip: TipState }) {
  if (!tip) return null;
  return (
    <div className="gtip" style={{ left: tip.x, top: tip.y }}>
      {tip.text}
    </div>
  );
}

/** Small "how to read this chart" box shown under a graph. */
export function HowToRead({ children }: { children: React.ReactNode }) {
  return (
    <p className="howto">
      <span className="howto-tag">How to read</span>
      {children}
    </p>
  );
}

/** Vertical bar chart with value labels and smooth hover tooltips. */
export function Bars({
  items,
  height = 260,
  fmt = (v: number) => `${v}`,
}: {
  items: { label: string; value: number; color?: string; title?: string }[];
  height?: number;
  fmt?: (v: number) => string;
}) {
  const { tip, wrap, move, leave } = useHoverTip();
  const W = 520;
  const pad = 30;
  const bottom = 46;
  const max = Math.max(...items.map((d) => d.value), 1);
  const bw = (W - 2 * pad) / items.length;
  const sy = (v: number) => height - bottom - (v / max) * (height - bottom - 14);
  return (
    <div className="chart-box" ref={wrap} style={{ position: "relative" }} onMouseLeave={leave}>
      <svg viewBox={`0 0 ${W} ${height}`}>
        {items.map((d, i) => {
          const x = pad + i * bw;
          const y = sy(d.value);
          const text = d.title ?? `${d.label}: ${fmt(d.value)}`;
          return (
            <g key={i}>
              <rect
                x={x + bw * 0.12}
                y={y}
                width={bw * 0.76}
                height={height - bottom - y}
                rx={4}
                fill={d.color ?? color(i)}
                className="mark"
                onMouseMove={(e) => move(e, text)}
              />
              <text x={x + bw / 2} y={y - 5} fill="var(--text)" fontSize={11} textAnchor="middle">
                {fmt(d.value)}
              </text>
              <text x={x + bw / 2} y={height - bottom + 16} fill="var(--muted)" fontSize={10} textAnchor="middle">
                {d.label}
              </text>
            </g>
          );
        })}
      </svg>
      <TipBox tip={tip} />
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
  const { tip, wrap, move, leave } = useHoverTip();
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
    <div className="chart-box" ref={wrap} style={{ position: "relative" }} onMouseLeave={leave}>
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
              className="mark"
              onMouseMove={(e) => move(e, `${binLo.toFixed(0)}–${binHi.toFixed(0)}${unit}: ${c} passengers`)}
            />
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
      <TipBox tip={tip} />
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
  const { tip, wrap, move, leave } = useHoverTip();
  const W = 520;
  const pad = 34;
  const lo = Math.min(...values, low);
  const hi = Math.max(...values, high);
  const sx = (v: number) => pad + ((v - lo) / (hi - lo || 1)) * (W - 2 * pad);
  const top = 22;
  const band = height - 44;
  const jit = (i: number) => top + (((i * 73 + 17) % 100) / 100) * band;
  return (
    <div className="chart-box" ref={wrap} style={{ position: "relative" }} onMouseLeave={leave}>
      <svg viewBox={`0 0 ${W} ${height}`}>
        <rect x={sx(low)} y={top} width={Math.max(sx(high) - sx(low), 1)} height={band} fill="#4C78A8" opacity={0.12} rx={6} />
        <line x1={sx(low)} y1={top} x2={sx(low)} y2={top + band} stroke="#4C78A8" strokeDasharray="4 4" opacity={0.6} />
        <line x1={sx(high)} y1={top} x2={sx(high)} y2={top + band} stroke="#4C78A8" strokeDasharray="4 4" opacity={0.6} />
        {values.map((v, i) => {
          const out = v < low || v > high;
          return (
            <circle
              key={i}
              cx={sx(v)}
              cy={jit(i)}
              r={3.5}
              fill={out ? "#fb7185" : "#4C78A8"}
              opacity={out ? 0.9 : 0.5}
              className="mark"
              onMouseMove={(e) => move(e, `fare $${v.toFixed(2)}${out ? "  —  outlier" : "  —  within range"}`)}
            />
          );
        })}
        <text x={sx(high) + 6} y={top + 12} fill="#4C78A8" fontSize={10}>normal range</text>
        <text x={W / 2} y={height - 6} fill="var(--muted)" fontSize={11} textAnchor="middle">
          value{unit ? ` (${unit})` : ""} →
        </text>
      </svg>
      <TipBox tip={tip} />
    </div>
  );
}

/** Scatter plot with two-class coloring and smooth hover tooltips. */
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
  const { tip, wrap, move, leave } = useHoverTip();
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
    <div className="chart-box" ref={wrap} style={{ position: "relative" }} onMouseLeave={leave}>
      <svg viewBox={`0 0 ${W} ${height}`}>
        <line x1={pad} y1={height - pad} x2={W - pad} y2={height - pad} stroke="var(--border)" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="var(--border)" />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={4}
            fill={p.c}
            opacity={0.72}
            className="mark"
            onMouseMove={(e) => move(e, p.title ?? `${xlabel} ${p.x}, ${ylabel} ${p.y}`)}
          />
        ))}
        <text x={W / 2} y={height - 8} fill="var(--muted)" fontSize={11} textAnchor="middle">
          {xlabel}
        </text>
        <text x={14} y={height / 2} fill="var(--muted)" fontSize={11} textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`}>
          {ylabel}
        </text>
      </svg>
      <TipBox tip={tip} />
    </div>
  );
}

/** A single vertical boxplot, scaled to its Tukey whiskers so a skewed column (fare) stays readable. */
export function MiniBox({
  label,
  stats,
  tone = "#a78bfa",
}: {
  label: string;
  stats: { min: number; q1: number; median: number; q3: number; max: number };
  tone?: string;
}) {
  const { tip, wrap, move, leave } = useHoverTip();
  const W = 120;
  const H = 240;
  const pad = 26;
  const { min, q1, median, q3, max } = stats;

  // Tukey fences: whiskers reach here, anything past them is an "outlier" region.
  const iqr = q3 - q1;
  const hiWhisk = Math.min(max, q3 + 1.5 * iqr);
  const loWhisk = Math.max(min, q1 - 1.5 * iqr);
  // View window centred on the box + whiskers (not the far-off max), so the box is never a sliver.
  const span = hiWhisk - loWhisk || 1;
  const lo = loWhisk - span * 0.12;
  const hi = hiWhisk + span * 0.12;
  const s = (v: number) => H - pad - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - 2 * pad);
  const cx = W / 2;
  const hasHiOutliers = max > hiWhisk + 1e-6;
  const hasLoOutliers = min < loWhisk - 1e-6;

  const summary = `${label}
max     ${max.toFixed(2)}
Q3      ${q3.toFixed(2)}
median  ${median.toFixed(2)}
Q1      ${q1.toFixed(2)}
min     ${min.toFixed(2)}`;

  return (
    <div className="chart-box" ref={wrap} style={{ padding: ".7rem", position: "relative" }} onMouseLeave={leave}>
      <p className="section-label" style={{ marginBottom: ".3rem", justifyContent: "center" }}>
        {label}
      </p>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {/* hover target covering the whole plot */}
        <rect x={0} y={0} width={W} height={H} fill="transparent" onMouseMove={(e) => move(e, summary)} />
        {/* whiskers */}
        <line x1={cx} y1={s(hiWhisk)} x2={cx} y2={s(loWhisk)} stroke="var(--muted)" strokeWidth={1.5} pointerEvents="none" />
        <line x1={cx - 12} y1={s(hiWhisk)} x2={cx + 12} y2={s(hiWhisk)} stroke="var(--muted)" pointerEvents="none" />
        <line x1={cx - 12} y1={s(loWhisk)} x2={cx + 12} y2={s(loWhisk)} stroke="var(--muted)" pointerEvents="none" />
        {/* box */}
        <rect x={cx - 24} y={s(q3)} width={48} height={Math.max(s(q1) - s(q3), 1)} rx={4} fill={tone} opacity={0.35} stroke={tone} pointerEvents="none" />
        <line x1={cx - 24} y1={s(median)} x2={cx + 24} y2={s(median)} stroke={tone} strokeWidth={2.5} pointerEvents="none" />
        {/* outlier indicators: data continues past the whisker */}
        {hasHiOutliers && (
          <g pointerEvents="none">
            <text x={cx} y={pad - 12} fill="#fb7185" fontSize={12} textAnchor="middle">▲</text>
            <text x={cx} y={pad - 2} fill="var(--muted)" fontSize={8} textAnchor="middle">{max.toFixed(0)}</text>
          </g>
        )}
        {hasLoOutliers && <text x={cx} y={H - 4} fill="#fb7185" fontSize={12} textAnchor="middle" pointerEvents="none">▼</text>}
      </svg>
      <p className="note" style={{ textAlign: "center", fontSize: ".72rem", marginTop: ".3rem" }}>
        {min.toFixed(2)} … {max.toFixed(2)}
      </p>
      <TipBox tip={tip} />
    </div>
  );
}
