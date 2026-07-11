"use client";

import { useJson, Bars, HowToRead } from "../lib/chart";
import { Tip } from "../lib/tip";

type Table = { cols: string[]; rows: (string | number)[][] };
type EncData = {
  original: Table;
  ordinal: Table & { legend: Record<string, Record<string, number>> };
  target: Table;
  onehot: Table;
  widths: Record<string, number>;
};

function DTable({ cols, rows }: { cols: string[]; rows: (string | number)[][] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="dtable">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EncodingTab() {
  const data = useJson<EncData>("/encoding.json");
  if (!data) return <p className="note">loading…</p>;

  const legendText = Object.entries(data.ordinal.legend)
    .map(([col, map]) => `${col}: ${Object.entries(map).map(([k, v]) => `${k}=${v}`).join(", ")}`)
    .join("  ·  ");

  const targetRows = data.target.rows.map((r) => r.map((v) => Number(v).toFixed(3)));

  return (
    <div className="demo">
      {/* Before */}
      <div className="subpanel">
        <p className="shead">
          Before — three text columns <Tip text="Models can only do math on numbers. These string columns must become numeric one way or another." />
        </p>
        <p className="sdesc">The first 6 passengers, three categorical features straight from the dataset.</p>
        <DTable cols={data.original.cols} rows={data.original.rows} />
      </div>

      {/* Ordinal */}
      <div className="subpanel">
        <p className="shead">
          1 · Ordinal encoding <Tip text="Each category becomes an integer. Compact (one column each), but it invents an order: the model may wrongly think category 2 is 'more than' category 1." />
        </p>
        <p className="sdesc">Every category is swapped for a number — same three columns, now numeric.</p>
        <DTable cols={data.ordinal.cols} rows={data.ordinal.rows} />
        <HowToRead>
          Compare this table to the one above, row for row: each word became a number using the fixed map{" "}
          <b>{legendText}</b>. Stays at <b>one column per feature</b>, but those numbers imply an ordering that may not
          exist — fine for truly ordered categories (small &lt; medium &lt; large), risky otherwise.
        </HowToRead>
      </div>

      {/* Target */}
      <div className="subpanel">
        <p className="shead">
          2 · Target encoding <Tip text="Each category becomes the average target for that group — here, the survival rate. Compact and informative, but it peeks at the label, so it MUST be fit on training data only." />
        </p>
        <p className="sdesc">Each category is replaced by its group’s mean survival rate (0–1).</p>
        <DTable cols={data.target.cols} rows={targetRows} />
        <HowToRead>
          Same shape as ordinal — <b>one column per feature</b> — but the number now carries meaning: a higher value means
          that category’s passengers survived more often. Powerful signal, but because it uses the target it will{" "}
          <b>leak</b> unless computed on the training split alone.
        </HowToRead>
      </div>

      {/* One-hot */}
      <div className="subpanel">
        <p className="shead">
          3 · One-hot encoding <Tip text="One new 0/1 column per category. No fake ordering, but the table gets wide fast when a feature has many categories." />
        </p>
        <p className="sdesc">Every distinct category becomes its own 0/1 column.</p>
        <DTable cols={data.onehot.cols} rows={data.onehot.rows} />
        <HowToRead>
          Read across a row: exactly <b>one 1</b> per original feature marks which category that passenger had; the rest are
          0. No invented ordering — but three text columns just became <b>{data.onehot.cols.length}</b> numeric ones.
        </HowToRead>
      </div>

      {/* Width comparison */}
      <div className="subpanel">
        <p className="shead">
          The trade-off in one chart <Tip text="This is the core decision: one-hot is safe but wide; ordinal and target stay narrow but carry assumptions or leakage risk." />
        </p>
        <Bars
          items={Object.entries(data.widths).map(([label, value]) => ({
            label,
            value,
            title: `${label}: ${value} columns`,
          }))}
          height={220}
        />
        <HowToRead>
          Each bar is <b>how many columns</b> that method produces from the same three features. One-hot balloons to{" "}
          <b>{data.widths["One-Hot"]}</b>; ordinal and target stay at <b>{data.widths["Ordinal"]}</b>. With high-cardinality
          features (hundreds of categories) that width gap is exactly why people reach for ordinal or target encoding.
        </HowToRead>
      </div>
    </div>
  );
}
