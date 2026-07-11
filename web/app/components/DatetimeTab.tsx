"use client";

import { useJson, Bars, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

type DtData = { by_hour: number[]; clock: { h: number; sin: number; cos: number }[] };

function Clock({ clock }: { clock: DtData["clock"] }) {
  const S = 300;
  const c = S / 2;
  const R = 110;
  return (
    <div className="chart-box">
      <svg viewBox={`0 0 ${S} ${S}`}>
        <circle cx={c} cy={c} r={R} fill="none" stroke="var(--border)" strokeDasharray="3 4" />
        {clock.map((p) => {
          const x = c + p.cos * R;
          const y = c - p.sin * R;
          return (
            <g key={p.h}>
              <circle cx={x} cy={y} r={12} fill="var(--panel-2)" stroke="var(--violet)">
                <title>{`hour ${p.h} → sin ${p.sin.toFixed(2)}, cos ${p.cos.toFixed(2)}`}</title>
              </circle>
              <text x={x} y={y + 3.5} fill="var(--text)" fontSize={9} textAnchor="middle">
                {p.h}
              </text>
            </g>
          );
        })}
        <text x={c} y={c} fill="var(--muted)" fontSize={11} textAnchor="middle">
          hour on a circle
        </text>
      </svg>
    </div>
  );
}

export default function DatetimeTab() {
  const data = useJson<DtData>("/datetime.json");
  if (!data) return <p className="note">loading…</p>;

  const peak = data.by_hour.indexOf(Math.max(...data.by_hour));

  return (
    <div className="demo">
      <div className="subpanel">
        <p className="shead">
          What one timestamp hides <Tip text="A raw datetime like '2019-03-23 20:21:09' is nearly useless to a model as one value. The signal lives in its parts — here, the hour of day." />
        </p>
        <p className="sdesc">Ride volume by hour, pulled out of the pickup timestamp with the pandas <code>.dt</code> accessor.</p>
        <Bars
          items={data.by_hour.map((v, h) => ({
            label: h % 3 === 0 ? `${h}` : "",
            value: v,
            title: `${h}:00–${h}:59 — ${v} rides`,
          }))}
          height={240}
          fmt={() => ""}
        />
        <HowToRead>
          The x-axis is <b>hour of day (0–23)</b>; each bar is <b>how many rides started</b> in that hour. Hover for the exact
          count. The shape is real structure a model can use — a quiet trough overnight and a clear peak around{" "}
          <b>{peak}:00</b> — none of which the raw timestamp string exposes on its own.
        </HowToRead>
      </div>

      <div className="subpanel">
        <p className="shead">
          Why time needs a circle <Tip text="Cyclical encoding maps a repeating value onto a circle with sine and cosine, so the model knows the end wraps back to the start." />
        </p>
        <p className="sdesc">The same 24 hours, placed on a circle by their (cos, sin) coordinates.</p>
        <div className="grid2">
          <Clock clock={data.clock} />
          <div className="callout">
            As plain integers, hour <b>23</b> and hour <b>0</b> look 23 apart — maximally distant, when they are actually
            adjacent. Encoding each hour as <b>(cos θ, sin θ)</b> places <b>23:00 right next to 00:00</b> on the ring, so the
            model finally sees the midnight wrap-around. Do the same for month, day-of-week, and any other repeating field.
          </div>
        </div>
        <HowToRead>
          Each node is one <b>hour</b>, positioned by its two new features — <b>cos on the x-axis, sin on the y-axis</b>. Hover
          a node to see those numbers. Walk the ring and notice it’s continuous: <b>23 sits beside 0</b>, exactly the
          adjacency a plain 0–23 integer throws away.
        </HowToRead>
      </div>
    </div>
  );
}
