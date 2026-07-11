"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { TABS } from "./models";
import { Tip } from "./lib/tip";

const load = (p: () => Promise<{ default: React.ComponentType }>) =>
  dynamic(p, { ssr: false, loading: () => <p className="note">loading…</p> });

const TAB_COMPONENTS: Record<string, React.ComponentType> = {
  about: load(() => import("./components/AboutTab")),
  missing: load(() => import("./components/MissingTab")),
  scaling: load(() => import("./components/ScalingTab")),
  encoding: load(() => import("./components/EncodingTab")),
  outliers: load(() => import("./components/OutliersTab")),
  datetime: load(() => import("./components/DatetimeTab")),
  pca: load(() => import("./components/PcaTab")),
  smote: load(() => import("./components/SmoteTab")),
};

export default function Home() {
  const [active, setActive] = useState(TABS[0].id);
  const tab = TABS.find((t) => t.id === active)!;
  const Comp = TAB_COMPONENTS[tab.id];

  return (
    <main className="wrap">
      <header className="hero">
        <h1>Data Preprocessing, Visualized</h1>
        <p>
          A model is only as good as the data it is trained on. This is an interactive, before-and-after tour of every
          core preprocessing step — from filling missing values to scaling, encoding, outlier detection and PCA —
          computed from the{" "}
          <a href="https://github.com/mwaskom/seaborn-data" target="_blank" rel="noreferrer">Titanic dataset</a>{" "}
          and running <strong>entirely in your browser</strong> from precomputed artifacts.
        </p>
        <span className="live">
          <b>●</b> static · no server · nothing leaves your machine
        </span>
      </header>

      <nav className="tabs" role="tablist" aria-label="Preprocessing steps">
        {TABS.map((t) => (
          <button key={t.id} className="tab" role="tab" aria-selected={t.id === active} onClick={() => setActive(t.id)}>
            {t.title}
          </button>
        ))}
      </nav>

      <section className="panel" role="tabpanel">
        <div className="panel-head">
          <div className="htitle">
            <h2>{tab.title}</h2>
            <Tip text={tab.help} />
          </div>
          <span className="chip">{tab.nb === "—" ? "Overview" : `Notebook ${tab.nb}`}</span>
        </div>
        <p className="panel-tagline">{tab.tagline}</p>
        {Comp ? <Comp /> : null}
      </section>

      <p className="footer">Built by Shivani Bokka · pandas / scikit-learn / imbalanced-learn · served client-side on Vercel</p>
    </main>
  );
}
