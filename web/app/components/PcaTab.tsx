"use client";

import { useJson, Bars, Scatter } from "../lib/chart";

type PcaData = {
  explained: number[];
  cumulative: number[];
  scatter: { x: number; y: number; s: number }[];
};

export default function PcaTab() {
  const data = useJson<PcaData>("/pca.json");
  if (!data) return <p className="note">loading…</p>;

  const keptPct = (data.cumulative[1] * 100).toFixed(0);

  return (
    <div className="demo">
      <div className="results">
        <div>
          <p className="section-label">Scree plot — variance explained by each of the 5 components</p>
          <Bars
            items={data.explained.map((v, i) => ({ label: `PC${i + 1}`, value: Math.round(v * 1000) / 10 }))}
            height={230}
            fmt={(v) => `${v}%`}
          />
          <p className="note" style={{ marginTop: ".6rem" }}>
            The first two components together retain <b>{keptPct}%</b> of the variance in the five numeric features —
            enough to plot the whole dataset in 2D below.
          </p>
        </div>

        <div>
          <p className="section-label">First two components, colored by survival</p>
          <Scatter
            points={data.scatter.map((p) => ({ x: p.x, y: p.y, c: p.s === 1 ? "#38bdf8" : "#fbbf24" }))}
            xlabel="PC1"
            ylabel="PC2"
          />
          <div className="legend">
            <span><i style={{ background: "#38bdf8" }} /> survived</span>
            <span><i style={{ background: "#fbbf24" }} /> did not survive</span>
          </div>
          <div className="callout" style={{ marginTop: "1rem" }}>
            PCA rotates the data onto new axes ordered by how much variance they capture. It never sees the labels — yet
            survivors and non-survivors already separate, a sign the numeric features carry real signal.
          </div>
        </div>
      </div>
    </div>
  );
}
