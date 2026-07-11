"use client";

import { useJson, Bars, Scatter, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

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
      <div className="subpanel">
        <p className="shead">
          How much does each component keep? <Tip text="PCA builds new axes (components) ordered by how much of the data's variance — its spread, i.e. information — each one captures." />
        </p>
        <p className="sdesc">Five numeric features become five ranked components. This scree plot shows their share of the variance.</p>
        <Bars
          items={data.explained.map((v, i) => ({
            label: `PC${i + 1}`,
            value: Math.round(v * 1000) / 10,
            title: `PC${i + 1}: ${(v * 100).toFixed(1)}% of variance (cumulative ${(data.cumulative[i] * 100).toFixed(1)}%)`,
          }))}
          height={230}
          fmt={(v) => `${v}%`}
        />
        <HowToRead>
          Each bar is a <b>component</b>; its height is the <b>percentage of total variance</b> it captures — bars always
          fall left-to-right because PCA orders them by importance. Hover for the running total. The first two together hold{" "}
          <b>{keptPct}%</b>, which is why we can throw away the other three and still plot the data faithfully in 2D below.
        </HowToRead>
      </div>

      <div className="subpanel">
        <p className="shead">
          Five features, drawn in two <Tip text="Every passenger placed using only PC1 and PC2. PCA never saw the survived label — yet the two groups still pull apart." />
        </p>
        <p className="sdesc">Each point is a passenger, colored by whether they survived.</p>
        <Scatter
          points={data.scatter.map((p) => ({
            x: p.x,
            y: p.y,
            c: p.s === 1 ? "#38bdf8" : "#fbbf24",
            title: `PC1 ${p.x}, PC2 ${p.y} — ${p.s === 1 ? "survived" : "did not survive"}`,
          }))}
          xlabel="PC1"
          ylabel="PC2"
        />
        <div className="legend">
          <span><i style={{ background: "#38bdf8" }} /> survived</span>
          <span><i style={{ background: "#fbbf24" }} /> did not survive</span>
        </div>
        <HowToRead>
          The axes are the two strongest components — not any original feature, but blends of all five. Position encodes a
          passenger’s overall profile; hover a point for its coordinates and outcome. Because the{" "}
          <span style={{ color: "#38bdf8" }}>blue</span> and <span style={{ color: "#fbbf24" }}>amber</span> clouds separate
          even though PCA ignored the label, the numeric features clearly carry real survival signal.
        </HowToRead>
      </div>
    </div>
  );
}
