"use client";

import { useState } from "react";
import { useJson, MiniBox } from "../lib/chart";

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
            <span className="lname">Column</span>
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
          Distribution of <b>{col}</b> after each scaler (each box self-scaled to its own range)
        </p>
        <div className="boxrow">
          {ORDER.map((name) => (
            <MiniBox key={name} label={name.replace("Scaler", "")} stats={series[name]} tone={TONE[name]} />
          ))}
        </div>
        <div className="callout" style={{ marginTop: "1.2rem" }}>
          The box shape — median line, quartile box, whiskers — is <b>identical</b> across every scaler. Only the axis
          numbers change: StandardScaler centers on 0, MinMax squeezes into [0, 1], RobustScaler ignores outliers by
          using the median and IQR. Scaling changes the <b>range</b>, never the <b>shape</b>.
        </div>
      </div>
    </div>
  );
}
