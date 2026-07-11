"use client";

import { useState } from "react";
import { useJson, Bars } from "../lib/chart";

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
  none: "Dropping rows keeps the true shape but throws away 177 passengers — a fifth of the data.",
  mean: "Mean imputation dumps every missing age into one bin at the average, spiking the distribution unnaturally.",
  median: "Median imputation is more robust to the fare-like skew, but still piles missing values onto a single value.",
  knn: "KNN fills each gap from the 5 most similar passengers, so the filled ages spread naturally instead of stacking.",
};

export default function MissingTab() {
  const data = useJson<MissData>("/missing.json");
  const [strat, setStrat] = useState("mean");
  if (!data) return <p className="note">loading…</p>;

  const s = data.age.strategies[strat];
  const edges = data.age.edges;
  const bars = s.hist.map((v, i) => ({ label: i % 2 === 0 ? `${Math.round(edges[i])}` : "", value: v }));

  return (
    <div className="demo">
      <div className="results">
        <div>
          <p className="section-label">Missing values per column (of {data.total_rows} rows)</p>
          <Bars
            items={data.counts.map((c) => ({ label: c.col, value: c.missing }))}
            height={230}
            fmt={(v) => `${v}`}
          />
        </div>

        <div>
          <p className="section-label">Impute the {data.age.n_missing} missing ages — pick a strategy</p>
          <div className="seg" style={{ marginBottom: "1rem" }}>
            {STRATS.map((o) => (
              <button key={o.id} aria-pressed={strat === o.id} onClick={() => setStrat(o.id)}>
                {o.label}
              </button>
            ))}
          </div>
          <Bars items={bars} height={260} fmt={(v) => (v ? `${v}` : "")} />
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
    </div>
  );
}
