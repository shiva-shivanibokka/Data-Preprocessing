"use client";

import { useJson, Scatter, Strip, Histogram, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

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
      <p className="section-label">
        Three methods, three answers — how many of {data.n_total} fares each one flags{" "}
        <Tip text="There is no single 'correct' number of outliers. Each method encodes a different definition of 'unusual', so they disagree — and that disagreement is the lesson." />
      </p>
      <div className="tiles">
        <div className="tile">
          <div className="v">{data.iqr.count}</div>
          <div className="k">IQR · {pct(data.iqr.count)} <span className="tiptag"><Tip text="Flags anything outside Q1−1.5×IQR to Q3+1.5×IQR. Distribution-free; generous on skewed data." /></span></div>
        </div>
        <div className="tile">
          <div className="v">{data.z.count}</div>
          <div className="k">z-score · {pct(data.z.count)} <span className="tiptag"><Tip text="Flags points more than 3 standard deviations from the mean. Assumes a bell curve, so it flags few points on skewed data like fare." /></span></div>
        </div>
        <div className="tile">
          <div className="v">{data.iso.count}</div>
          <div className="k">Isolation · {pct(data.iso.count)} <span className="tiptag"><Tip text="A model that isolates points unusual across MULTIPLE features at once. Contamination was set to 5%." /></span></div>
        </div>
      </div>

      {/* IQR */}
      <div className="subpanel" style={{ marginTop: "1.5rem" }}>
        <p className="shead">
          1 · IQR rule <Tip text="The same rule that draws a boxplot's whiskers. Simple, robust, no distribution assumption." />
        </p>
        <p className="sdesc">Every fare laid on a line; the shaded band is the “normal” range, points outside are outliers.</p>
        <Strip values={data.fare_sample} low={data.iqr.low} high={data.iqr.high} height={150} unit="$" />
        <HowToRead>
          Each dot is one passenger’s <b>fare</b>, spread out vertically only so they don’t overlap. The shaded band is the
          IQR-normal range <b>[{data.iqr.low}, {data.iqr.high}]</b>; every <span style={{ color: "#fb7185" }}>red</span> dot
          past its right edge is a flagged outlier. Hover a dot for its exact fare. Because fare is right-skewed, this rule
          flags a lot — <b>{data.iqr.count}</b> in total.
        </HowToRead>
      </div>

      {/* z-score */}
      <div className="subpanel">
        <p className="shead">
          2 · z-score rule <Tip text="Standardize, then flag |z| > 3. Only sensible when the data is roughly normal." />
        </p>
        <p className="sdesc">The fare distribution with the 3-standard-deviation cutoff drawn in.</p>
        <Histogram values={data.fare_sample} bins={26} cutoff={data.z.cut} cutoffLabel={`cutoff $${data.z.cut}`} unit="$" height={240} />
        <HowToRead>
          The x-axis is <b>fare</b>, each bar is <b>how many passengers</b> paid in that range. The red dashed line sits at{" "}
          <b>mean + 3·std = ${data.z.cut}</b>; only the <span style={{ color: "#fb7185" }}>red</span> bars beyond it are
          flagged. Because it assumes a bell curve, it’s far stricter here — just <b>{data.z.count}</b> outliers.
        </HowToRead>
      </div>

      {/* IsolationForest */}
      <div className="subpanel">
        <p className="shead">
          3 · IsolationForest <Tip text="Looks at age and fare TOGETHER, so it can flag combinations that are odd even when neither value alone is extreme." />
        </p>
        <p className="sdesc">Now with two features at once — age on the x-axis, fare on the y-axis.</p>
        <Scatter
          points={data.iso.points.map((p) => ({
            x: p.age,
            y: p.fare,
            c: p.out ? "#fb7185" : "#4C78A8",
            title: `age ${p.age}, fare $${p.fare}${p.out ? " — outlier" : ""}`,
          }))}
          xlabel="age"
          ylabel="fare"
        />
        <div className="legend">
          <span><i style={{ background: "#4C78A8" }} /> inlier</span>
          <span><i style={{ background: "#fb7185" }} /> outlier</span>
        </div>
        <HowToRead>
          Each dot is one passenger placed by <b>both</b> age and fare; hover for the values. The single-column rules above
          could only look left-to-right or bottom-to-top. This one flags <span style={{ color: "#fb7185" }}>red</span> points
          that are odd <b>jointly</b> — e.g. a very young passenger paying a very high fare — which IQR and z-score can miss.
        </HowToRead>
      </div>

      <div className="callout">
        The takeaway isn’t “which method is right” — it’s that <b>“outlier” is a choice of definition</b>. Detect first, then
        decide per case whether to cap, drop, or keep. Never delete a point just because a rule lit up.
      </div>
    </div>
  );
}
