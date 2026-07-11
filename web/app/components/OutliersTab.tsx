"use client";

import { useJson, Scatter } from "../lib/chart";

type OutData = {
  fare_sample: number[];
  iqr: { low: number; high: number; count: number };
  z: { cut: number; count: number };
  iso: { count: number; points: { age: number; fare: number; out: boolean }[] };
  n_total: number;
};

export default function OutliersTab() {
  const data = useJson<OutData>("/outliers.json");
  if (!data) return <p className="note">loading…</p>;

  const pct = (n: number) => `${((100 * n) / data.n_total).toFixed(1)}%`;

  return (
    <div className="demo">
      <div className="results">
        <div>
          <p className="section-label">How many of {data.n_total} fares each method flags as outliers</p>
          <div className="tiles">
            <div className="tile">
              <div className="v">{data.iqr.count}</div>
              <div className="k">IQR · {pct(data.iqr.count)}</div>
            </div>
            <div className="tile">
              <div className="v">{data.z.count}</div>
              <div className="k">z-score · {pct(data.z.count)}</div>
            </div>
            <div className="tile">
              <div className="v">{data.iso.count}</div>
              <div className="k">IsolationForest · {pct(data.iso.count)}</div>
            </div>
          </div>
          <p className="note" style={{ marginTop: ".7rem" }}>
            IQR flags anything outside <b>[{data.iqr.low}, {data.iqr.high}]</b>; z-score flags fares above{" "}
            <b>{data.z.cut}</b>. They disagree because fare is strongly right-skewed — the generous IQR rule catches far
            more than the normality-assuming z-score.
          </p>
        </div>

        <div>
          <p className="section-label">IsolationForest — joint age × fare outliers</p>
          <Scatter
            points={data.iso.points.map((p) => ({ x: p.age, y: p.fare, c: p.out ? "#fb7185" : "#4C78A8" }))}
            xlabel="age"
            ylabel="fare"
          />
          <div className="legend">
            <span><i style={{ background: "#4C78A8" }} /> inlier</span>
            <span><i style={{ background: "#fb7185" }} /> outlier</span>
          </div>
          <div className="callout" style={{ marginTop: "1rem" }}>
            IQR and z-score look at one column at a time. IsolationForest flags rows that are unusual{" "}
            <b>jointly</b> — a young passenger paying a very high fare — which neither single-column rule can see.
          </div>
        </div>
      </div>
    </div>
  );
}
