"use client";

import { useState } from "react";
import { useJson, MiniBox, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

type Box = { min: number; q1: number; median: number; q3: number; max: number; sample: number[] };
type ScaleData = Record<string, Record<string, Box>>;

const ORDER = ["Original", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"];
const TONE: Record<string, string> = {
  Original: "#9aa7c7",
  StandardScaler: "#22d3ee",
  MinMaxScaler: "#a78bfa",
  RobustScaler: "#f472b6",
  MaxAbsScaler: "#a3e635",
};

export default function ScalingTab() {
  const data = useJson<ScaleData>("/scaling.json");
  const [col, setCol] = useState("age");
  if (!data) return <p className="note">loading…</p>;

  const series = data[col];
  return (
    <div className="demo">
      <div className="control-row">
        <div className="field" style={{ flex: "0 0 auto" }}>
          <label>
            <span className="lname">
              Column <Tip text="Try both. age is a tidy 0–80 range; fare is skewed with big outliers, which is where RobustScaler earns its name." />
            </span>
          </label>
          <div className="seg">
            {Object.keys(data).map((c) => (
              <button key={c} aria-pressed={col === c} onClick={() => setCol(c)}>
                {c}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="results">
        <p className="section-label">
          Distribution of <b>{col}</b>&nbsp;after each scaler{" "}
          <Tip text="Each panel is one scaler applied to the same column. The boxes look identical because scaling is just multiply-and-shift — it never reorders or reshapes the data." />
        </p>
        <div className="boxrow">
          {ORDER.map((name) => (
            <MiniBox key={name} label={name.replace("Scaler", "")} stats={series[name]} tone={TONE[name]} />
          ))}
        </div>

        <HowToRead>
          Each panel is a <b>box-and-whisker plot</b> of the same column after a different scaler. The <b>box</b> spans the
          middle 50% of values (Q1 to Q3), the <b>line across it</b> is the median, and the <b>whiskers</b> reach the min and
          max. Hover any box to see its exact five numbers. Crucially, <b>every panel is scaled to its own range</b> — read the
          min…max printed under each. That’s the whole point: the box <b>shape</b> is identical everywhere, but the numbers on
          the axis differ.
        </HowToRead>

        <div className="callout" style={{ marginTop: "1.2rem" }}>
          Why it matters: models that measure distance (k-NN, SVM) or use gradients (neural nets) let big-numbered features
          drown out small ones. Scaling puts them on equal footing. <b>StandardScaler</b> centers on 0 (mean 0, std 1),{" "}
          <b>MinMax</b> squeezes into [0, 1], and <b>RobustScaler</b> uses the median and IQR so a few extreme{" "}
          {col === "fare" ? "fares" : "ages"} can’t distort the scale.
        </div>
      </div>
    </div>
  );
}
