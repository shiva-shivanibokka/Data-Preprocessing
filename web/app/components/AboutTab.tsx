"use client";

export default function AboutTab() {
  return (
    <div className="about">
      <div className="callout" style={{ marginBottom: "1.4rem" }}>
        Every tab here is a <span className="k">precomputed before/after</span> of one preprocessing step. The numbers
        come from a Python export script that runs the exact code in the companion notebooks — the browser just draws
        the results. No server, no data leaves your machine.
      </div>

      <h3><span className="num">01</span>Pandas &amp; NumPy fundamentals</h3>
      <p>
        The hand-rolled toolkit: exploring data, handling missing values with <span className="k">fillna / dropna</span>,
        removing duplicates, fixing data types, reshaping and engineering features. See the{" "}
        <b>Missing Values</b> tab.
      </p>

      <h3><span className="num">02</span>The scikit-learn way</h3>
      <p>
        The same goals, done with <span className="k">sklearn transformers</span> that fit on training data and drop
        into pipelines: imputers, scalers, encoders, PCA, and SMOTE for class imbalance. See the{" "}
        <b>Scaling</b>, <b>Encoding</b>, <b>PCA</b> and <b>Class Imbalance</b> tabs.
      </p>

      <h3><span className="num">03</span>Advanced concepts</h3>
      <p>
        The techniques a &ldquo;complete&rdquo; reference usually skips: <span className="k">outlier detection</span>{" "}
        (IQR, z-score, IsolationForest), <span className="k">datetime feature extraction</span> with cyclical encoding,
        and leakage-safe target encoding. See the <b>Outliers</b> and <b>Datetime</b> tabs.
      </p>

      <h3><span className="num">★</span>The one rule that ties it together</h3>
      <p>
        <b>Never fit a transformer on the test set.</b> Learn fill values, scales, and encodings from training data
        only, then apply them to test data. Fitting on everything leaks information and makes your model look better
        than it really is. Every demo here respects that split.
      </p>
    </div>
  );
}
