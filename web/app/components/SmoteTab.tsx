"use client";

import { useJson, Bars } from "../lib/chart";

type SmoteData = {
  before: { died: number; survived: number };
  after: { died: number; survived: number };
};

export default function SmoteTab() {
  const data = useJson<SmoteData>("/smote.json");
  if (!data) return <p className="note">loading…</p>;

  const mk = (d: { died: number; survived: number }) => [
    { label: "did not survive", value: d.died, color: "#fbbf24" },
    { label: "survived", value: d.survived, color: "#38bdf8" },
  ];

  return (
    <div className="demo">
      <div className="results">
        <div className="grid2">
          <div>
            <p className="section-label">Training set — before SMOTE</p>
            <Bars items={mk(data.before)} height={240} fmt={(v) => `${v}`} />
          </div>
          <div>
            <p className="section-label">Training set — after SMOTE</p>
            <Bars items={mk(data.after)} height={240} fmt={(v) => `${v}`} />
          </div>
        </div>
        <div className="callout">
          The minority class was lifted from <b>{data.before.survived}</b> to <b>{data.after.survived}</b> synthetic-plus-real
          samples, matching the majority. SMOTE interpolates new points between real minority neighbors rather than copying
          them. Crucially, it runs on the <b>training set only</b> — the test set stays imbalanced so evaluation reflects the
          real-world distribution.
        </div>
      </div>
    </div>
  );
}
