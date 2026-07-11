"use client";

import { useJson, Bars } from "../lib/chart";

type Table = { cols: string[]; rows: (string | number)[][] };
type EncData = { original: Table; onehot: Table; widths: Record<string, number> };

function DTable({ t }: { t: Table }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="dtable">
        <thead>
          <tr>
            {t.cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {t.rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j}>{String(v)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EncodingTab() {
  const data = useJson<EncData>("/encoding.json");
  if (!data) return <p className="note">loading…</p>;

  return (
    <div className="demo">
      <div className="results">
        <div>
          <p className="section-label">Before — 3 text columns (first 6 rows)</p>
          <DTable t={data.original} />
        </div>
        <div>
          <p className="section-label">After one-hot encoding — {data.onehot.cols.length} numeric columns</p>
          <DTable t={data.onehot} />
        </div>
        <div>
          <p className="section-label">Column count by encoding method</p>
          <Bars
            items={Object.entries(data.widths).map(([label, value]) => ({ label, value }))}
            height={220}
            fmt={(v) => `${v}`}
          />
          <div className="callout" style={{ marginTop: "1rem" }}>
            One-hot encoding creates a column for <b>every category</b>, so a handful of text columns balloon into many.
            Ordinal and target encoding keep each feature to a <b>single</b> column — far leaner for high-cardinality
            features, at the cost of imposing an order (ordinal) or risking leakage (target).
          </div>
        </div>
      </div>
    </div>
  );
}
