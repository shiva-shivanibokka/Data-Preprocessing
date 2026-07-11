"use client";

import { useState } from "react";
import { useJson, Bars, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

type Strat = { hist: number[]; fill_value: number | null };
type MissData = {
  counts: { col: string; missing: number; pct: number }[];
  total_rows: number;
  age: { edges: number[]; n_missing: number; strategies: Record<string, Strat> };
};

const STRATS: { id: string; label: string }[] = [
  { id: "none", label: "drop (observed only)" },
  { id: "mean", label: "mean" },
  { id: "median", label: "median" },
  { id: "knn", label: "KNN (k=5)" },
];

const BLURB: Record<string, string> = {
  none: "Dropping keeps the true shape but throws away 177 passengers — about a fifth of the data. Notice every bar is the same as the raw data; nothing was invented.",
  mean: "Watch the 25–30 bar rocket from 106 up to 283. That jump of ~177 is every missing age being stacked onto the single average value (29.7) — the distribution now has an unnatural spike.",
  median: "Same story as mean: the 25–30 bar spikes to 283 because all 177 gaps are filled with one number (median 28). Robust to skew, but still an artificial pile-up.",
  knn: "KNN fills each gap from the 5 most similar passengers, so the 177 values spread across several bars instead of piling onto one — the shape stays far more natural.",
};

export default function MissingTab() {
  const data = useJson<MissData>("/missing.json");
  const [strat, setStrat] = useState("mean");
  if (!data) return <p className="note">loading…</p>;

  const s = data.age.strategies[strat];
  const edges = data.age.edges;
  const bars = s.hist.map((v, i) => ({
    label: i % 2 === 0 ? `${Math.round(edges[i])}` : "",
    value: v,
    title: `Age ${Math.round(edges[i])}–${Math.round(edges[i + 1] ?? edges[i] + 5)}: ${v} passengers`,
  }));

  return (
    <div className="demo">
      {/* Box 1 — where the holes are */}
      <div className="subpanel">
        <p className="shead">
          Where are the holes? <Tip text="isna().sum() counts how many cells are empty in each column. deck is missing for 77% of passengers — often too sparse to keep." />
        </p>
        <p className="sdesc">Every column that has at least one missing value, out of {data.total_rows} total rows.</p>
        <Bars
          items={data.counts.map((c) => ({
            label: c.col,
            value: c.missing,
            title: `${c.col}: ${c.missing} missing (${c.pct}%)`,
          }))}
          height={230}
        />
        <HowToRead>
          Each bar is <b>one column</b>; its height is <b>how many rows are missing</b> that value. Hover any bar for the
          exact count and percentage. <b>deck</b> towers over the rest — missing for most passengers — while <b>age</b> is
          missing for {data.age.n_missing}.
        </HowToRead>
      </div>

      {/* Box 2 — fixing the age gaps */}
      <div className="subpanel">
        <p className="shead">
          Filling the {data.age.n_missing} missing ages <Tip text="Imputation = filling gaps with a best guess. The choice of guess changes the shape of your data, which changes what your model learns." />
        </p>
        <p className="sdesc">
          Pick a strategy and watch what it does to the <b>age</b> distribution. Same 891 passengers every time — only the
          177 previously-missing ages change.
        </p>
        <div className="seg" style={{ marginBottom: "1rem" }}>
          {STRATS.map((o) => (
            <button key={o.id} aria-pressed={strat === o.id} onClick={() => setStrat(o.id)}>
              {o.label}
            </button>
          ))}
        </div>
        <Bars items={bars} height={260} fmt={(v) => (v ? `${v}` : "")} />
        <HowToRead>
          The x-axis is <b>age in 5-year bins</b>; each bar’s height is <b>how many passengers fall in that range</b>. Hover a
          bar for its exact range and count. The tallest raw bar is only ~114 (the natural 20–25 peak) — so when a single bar
          shoots to <b>283</b>, that spike <b>is</b> the 177 missing ages all being dumped into one value.
        </HowToRead>
        <div className="callout" style={{ marginTop: "1rem" }}>
          {BLURB[strat]}
          {s.fill_value !== null && (
            <>
              {" "}
              Every gap was filled with <b>{s.fill_value}</b>.
            </>
          )}
        </div>
      </div>
    </div>
  );
}
