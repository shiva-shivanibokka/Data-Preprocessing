"use client";

import { useJson, Bars, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

type SmoteData = {
  before: { died: number; survived: number };
  after: { died: number; survived: number };
};

export default function SmoteTab() {
  const data = useJson<SmoteData>("/smote.json");
  if (!data) return <p className="note">loading…</p>;

  const mk = (d: { died: number; survived: number }) => [
    { label: "did not survive", value: d.died, color: "#fbbf24", title: `did not survive: ${d.died}` },
    { label: "survived", value: d.survived, color: "#38bdf8", title: `survived: ${d.survived}` },
  ];
  const added = data.after.survived - data.before.survived;

  return (
    <div className="demo">
      <div className="subpanel">
        <p className="shead">
          The imbalance, before and after <Tip text="SMOTE (Synthetic Minority Over-sampling) grows the smaller class by inventing new points between real neighbors, until the classes match." />
        </p>
        <p className="sdesc">Class counts in the training set, before and after applying SMOTE.</p>
        <div className="grid2">
          <div>
            <p className="section-label">Before SMOTE</p>
            <Bars items={mk(data.before)} height={240} />
          </div>
          <div>
            <p className="section-label">After SMOTE</p>
            <Bars items={mk(data.after)} height={240} />
          </div>
        </div>
        <HowToRead>
          Two bars per chart — one per class; height is the <b>number of training passengers</b>. Hover for exact counts. On
          the left the <span style={{ color: "#38bdf8" }}>survived</span> bar is clearly shorter; on the right SMOTE has added{" "}
          <b>{added}</b> synthetic survivors so both bars match. The two charts share the same story: minority class lifted up
          to balance the majority.
        </HowToRead>
        <div className="callout" style={{ marginTop: "1rem" }}>
          Why balance at all? A model trained on 62% / 38% can score well just by leaning toward “did not survive”. Balancing
          forces it to actually learn the minority class. One rule, though: SMOTE runs on the <b>training set only</b> — the
          test set stays imbalanced so your evaluation reflects the real world.
        </div>
      </div>
    </div>
  );
}
