"use client";

import { useJson, Bars } from "../lib/chart";

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
          // cos -> x, sin -> y (screen y is inverted)
          const x = c + p.cos * R;
          const y = c - p.sin * R;
          return (
            <g key={p.h}>
              <circle cx={x} cy={y} r={12} fill="var(--panel-2)" stroke="var(--violet)" />
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

  return (
    <div className="demo">
      <div className="results">
        <div>
          <p className="section-label">Ride volume by hour (NYC taxis — extracted from the pickup timestamp)</p>
          <Bars
            items={data.by_hour.map((v, h) => ({ label: h % 3 === 0 ? `${h}` : "", value: v }))}
            height={240}
            fmt={() => ""}
          />
        </div>

        <div className="grid2">
          <div>
            <Clock clock={data.clock} />
          </div>
          <div className="callout">
            As plain integers, hour <b>23</b> and hour <b>0</b> look 23 apart — maximally distant, when they are
            actually adjacent. Encoding each hour as{" "}
            <b>(cos θ, sin θ)</b> on a circle places <b>23:00 right next to 00:00</b>, so the model finally sees the
            midnight wrap-around. Do the same for month, day-of-week, and any other cyclical field.
          </div>
        </div>
      </div>
    </div>
  );
}
