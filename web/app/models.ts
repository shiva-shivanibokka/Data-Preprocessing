// Single source of truth for the explainer tabs. Each maps to a preprocessing step.
export type Tab = {
  id: string;
  nb: string;
  title: string;
  tagline: string;
  help: string;
};

export const TABS: Tab[] = [
  {
    id: "about", nb: "—", title: "Overview",
    tagline: "What preprocessing is, why it decides how good your model can get, and how these demos map to the notebooks.",
    help: "Start here. Every other tab is a precomputed before/after of one preprocessing step, all built from the Titanic dataset.",
  },
  {
    id: "missing", nb: "01 · 02", title: "Missing Values",
    tagline: "See which columns are full of holes, then watch each imputation strategy reshape the age distribution.",
    help: "Missing data breaks most models. You can drop it or fill it (impute). Mean/median imputation spikes one bin; KNN keeps the shape more natural.",
  },
  {
    id: "scaling", nb: "02", title: "Scaling",
    tagline: "Same column, five scalers. The shape stays identical — only the numbers on the axis change.",
    help: "Scaling puts features on a comparable range so distance- and gradient-based models behave. Each scaler is fit on training data only.",
  },
  {
    id: "encoding", nb: "02", title: "Encoding",
    tagline: "Turn text categories into numbers — and see why one-hot encoding can explode your column count.",
    help: "Models need numbers, not strings. One-hot makes one column per category; ordinal/target encoding keep it to a single column.",
  },
  {
    id: "outliers", nb: "03", title: "Outliers",
    tagline: "Three detectors — IQR, z-score, IsolationForest — disagree on purpose. Compare what each one flags.",
    help: "Outliers can be errors to drop or rare-but-real events to keep. Single-column rules miss joint outliers that IsolationForest catches.",
  },
  {
    id: "datetime", nb: "03", title: "Datetime",
    tagline: "A raw timestamp is dead weight until you extract its parts — and encode cyclical time on a circle.",
    help: "Hour 23 and hour 0 are one hour apart. Sine/cosine encoding places them next to each other so the model sees the wrap-around.",
  },
  {
    id: "pca", nb: "02", title: "PCA",
    tagline: "Squeeze five numeric features into two components and plot survivors against non-survivors.",
    help: "PCA rotates the data onto axes of maximum variance. The scree plot shows how much information each component keeps.",
  },
  {
    id: "smote", nb: "02", title: "Class Imbalance",
    tagline: "62% died, 38% survived. Watch SMOTE synthesize minority samples — on the training set only.",
    help: "Imbalanced classes bias a model toward the majority. SMOTE creates synthetic minority examples. Never apply it to the test set.",
  },
];
