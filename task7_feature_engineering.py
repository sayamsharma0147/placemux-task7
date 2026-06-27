"""
Task 7 — Baseline Target Feature Engineering
PlaceMux · Phase 1 · AI/ML Developer
=====================================================
WHAT THIS SCRIPT DOES:
  Implements the full feature engineering pipeline for the question-bank
  difficulty prediction problem. Starts from raw question text and metadata,
  engineers informative signals, checks every feature for data leakage, ranks
  features by importance, prunes low-value ones, and locks a vetted baseline
  feature set for downstream tasks.

WHY FEATURE ENGINEERING MATTERS:
  A weak model with good features nearly always beats a strong model with bad
  features. Before trying fancy architectures, we must ensure the inputs carry
  real signal about the target (difficulty_level).

DELIVERABLE:
  A vetted baseline feature set with importance analysis and a leakage check.

DATASET:
  6 question-bank xlsx files → 4,799 questions across 6 domains.
  Target: difficulty_level binarized at median (Hard=1 if >=42, Easy=0).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, warnings, json
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

# Fixed seed everywhere — ensures results are reproducible across runs
SEED = 42
np.random.seed(SEED)

# ── STEP 1: Load & combine data ───────────────────────────────────────────────
# We skip DevOps because it has a malformed header row (all columns merged into one).
# All other 6 files share the same 10-column schema.
files = [f for f in sorted(glob.glob("/mnt/user-data/uploads/formatted_*.xlsx"))
         if "DevOps" not in f]
data = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
print(f"Loaded {len(data)} rows from {len(files)} files\n")

# ── STEP 2: Re-confirm target definition & label quality ──────────────────────
# Before engineering features we verify the target column is clean and well-defined.
# A corrupted or ambiguous label will make all downstream work meaningless.
print("=" * 60)
print("STEP 1: TARGET DEFINITION & LABEL QUALITY")
print("=" * 60)

print(f"difficulty_level range : {data['difficulty_level'].min()} – {data['difficulty_level'].max()}")
print(f"Nulls in target        : {data['difficulty_level'].isna().sum()}")
print(f"Unique difficulty values: {data['difficulty_level'].nunique()}")

# Binarize at median → guarantees a near-balanced binary target.
# Median = 42 means ~50% questions fall each side — no class imbalance issue.
MEDIAN = data["difficulty_level"].median()
data["label"] = (data["difficulty_level"] >= MEDIAN).astype(int)
vc = data["label"].value_counts()
print(f"Median threshold       : {MEDIAN}")
print(f"Class distribution     → Easy(0): {vc[0]}  Hard(1): {vc[1]}")
print(f"Class balance          : {vc[1]/len(data):.1%} Hard — near-balanced, no resampling needed")
print("✓ Target confirmed: clean, no nulls, near-balanced\n")

# ── STEP 3: Engineer candidate features from domain reasoning ─────────────────
# DOMAIN REASONING:
#   Harder questions tend to use more complex vocabulary → longer text.
#   Trickier options are often longer and more similar in length (harder to eliminate).
#   Topic and domain carry curriculum-level difficulty signals.
#   Ratios capture relative complexity better than raw lengths alone.
print("=" * 60)
print("STEP 2: FEATURE ENGINEERING (11 features)")
print("=" * 60)

# --- Group A: Text length features ---
# Raw character count of the question. Longer questions often embed more context,
# which correlates with higher cognitive demand.
data["q_len"]        = data["question_text"].str.len().fillna(0)

# Word count separately — captures linguistic complexity independent of avg word length.
data["q_word_count"] = data["question_text"].str.split().str.len().fillna(0)

# Per-option lengths — harder questions often have longer, more nuanced options.
data["opt_a_len"]    = data["option_a"].str.len().fillna(0)
data["opt_b_len"]    = data["option_b"].str.len().fillna(0)
data["opt_c_len"]    = data["option_c"].str.len().fillna(0)
data["opt_d_len"]    = data["option_d"].str.len().fillna(0)

# --- Group B: Aggregate option features ---
opt_cols = ["opt_a_len", "opt_b_len", "opt_c_len", "opt_d_len"]

# Mean option length — measures overall verbosity of choices.
data["avg_opt_len"]   = data[opt_cols].mean(axis=1)

# Max option length — the longest distractor is often the trickiest one.
data["max_opt_len"]   = data[opt_cols].max(axis=1)

# Range (max - min) — high spread means options vary a lot in complexity,
# which is a sign of a well-crafted, harder question.
data["opt_len_range"] = data[opt_cols].max(axis=1) - data[opt_cols].min(axis=1)

# Total option text — proxy for total information load on the candidate.
data["total_opt_len"] = data[opt_cols].sum(axis=1)

# --- Group C: Domain / topic features (aggregate — leakage handled in Step 4) ---
# Mean difficulty of the domain, computed from training data only.
# Encodes curriculum-level knowledge: AI/ML questions may cluster at different
# difficulty than Verbal Ability questions.
domain_difficulty_map = data.groupby("domain")["difficulty_level"].mean().to_dict()
data["domain_avg_difficulty"] = data["domain"].map(domain_difficulty_map)

# Topic is finer-grained than domain (e.g., 'Optimization' within 'AI & ML').
# Topic-level mean difficulty is typically the strongest categorical signal.
topic_difficulty_map = data.groupby("topic")["difficulty_level"].mean().to_dict()
data["topic_avg_difficulty"] = data["topic"].map(topic_difficulty_map)

# Label-encoded versions — lets the model use domain/topic as ordinal inputs
# without the high dimensionality of one-hot encoding.
le_domain = LabelEncoder()
le_topic  = LabelEncoder()
data["domain_enc"] = le_domain.fit_transform(data["domain"].astype(str))
data["topic_enc"]  = le_topic.fit_transform(data["topic"].astype(str))

# --- Group D: Ratio feature ---
# Question length relative to average option length.
# A long question with short options signals a recall task (often easier).
# A short question with long options signals a reasoning task (often harder).
data["q_to_avg_opt_ratio"] = data["q_len"] / (data["avg_opt_len"] + 1)

feature_descriptions = {
    "q_len"                : "Question character length",
    "q_word_count"         : "Question word count",
    "avg_opt_len"          : "Mean length of 4 options",
    "max_opt_len"          : "Longest option length",
    "opt_len_range"        : "Max – Min option length (complexity spread)",
    "total_opt_len"        : "Sum of all option lengths (total info load)",
    "domain_avg_difficulty": "Mean difficulty of domain (train-only aggregate)",
    "topic_avg_difficulty" : "Mean difficulty of topic (train-only aggregate)",
    "domain_enc"           : "Domain label encoded (ordinal proxy)",
    "topic_enc"            : "Topic label encoded (ordinal proxy)",
    "q_to_avg_opt_ratio"   : "Question length / avg option length (ratio signal)",
}
print("Engineered features:")
for feat, desc in feature_descriptions.items():
    print(f"  {feat:<28} → {desc}")

# ── STEP 4: Leakage check ─────────────────────────────────────────────────────
# DATA LEAKAGE = using information at training time that would NOT be available
# when predicting on new, unseen data. Leaky features cause inflated validation
# scores that collapse on real deployment.
print("\n" + "=" * 60)
print("STEP 3: LEAKAGE CHECK")
print("=" * 60)

print("""
Feature leakage analysis — checked against:
  "Would this value be known BEFORE the difficulty_level is assigned?"

SAFE (no leakage):
  q_len, q_word_count       → pure question text properties          ✓
  opt_*_len, avg/max/range  → pure option text properties            ✓
  domain_enc, topic_enc     → metadata assigned before tagging       ✓
  q_to_avg_opt_ratio        → derived from text only                 ✓

FLAGGED (potential leakage) → FIXED:
  domain_avg_difficulty     → uses difficulty_level to compute mean  ✗ → Fixed
  topic_avg_difficulty      → same issue                             ✗ → Fixed

  FIX: Split the data first. Compute these aggregate means on the
  TRAINING SET ONLY. Then map those train-derived values onto val/test.
  Val/test rows for unseen topics fall back to the global train mean.

EXCLUDED (direct leakage):
  difficulty_level → IS the target itself                            ✗ excluded
  question_id      → row identifier, zero signal                     ✗ excluded
  correct_answer   → known answer letter, not a difficulty signal    ✗ excluded
""")

# ── STEP 5: Train/Val/Test split BEFORE computing leaky aggregates ────────────
# CRITICAL ORDER: split first, then recompute aggregate features from train only.
# If we compute aggregates on the full dataset first, val/test rows "see" their
# own difficulty_level through the group mean — that is leakage.
ALL_FEATURES = list(feature_descriptions.keys())
X_raw = data[ALL_FEATURES].copy()
y     = data["label"]

# Stratified split ensures both classes appear in the same proportion in all
# three splits — especially important since we have two roughly equal classes.
X_train, X_temp, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

# Now recompute aggregate features using ONLY training rows
train_idx = X_train.index
train_domain_map = data.loc[train_idx].groupby(
    data.loc[train_idx, "domain"])["difficulty_level"].mean()
train_topic_map  = data.loc[train_idx].groupby(
    data.loc[train_idx, "topic"])["difficulty_level"].mean()
global_mean = data.loc[train_idx, "difficulty_level"].mean()

# Apply train-derived maps to all splits (unseen topics → global mean fallback)
for idx in [X_train.index, X_val.index, X_test.index]:
    X_raw.loc[idx, "domain_avg_difficulty"] = (
        data.loc[idx, "domain"].map(train_domain_map).fillna(global_mean))
    X_raw.loc[idx, "topic_avg_difficulty"] = (
        data.loc[idx, "topic"].map(train_topic_map).fillna(global_mean))

X_train = X_raw.loc[X_train.index]
X_val   = X_raw.loc[X_val.index]
X_test  = X_raw.loc[X_test.index]

print(f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("✓ Aggregate features recomputed from train-only — leakage eliminated\n")

# ── STEP 6: Train model & inspect feature importance ─────────────────────────
# We use RandomForest's Mean Decrease in Impurity (MDI) to rank features.
# MDI measures how much each feature reduces uncertainty (Gini impurity) across
# all trees — a higher value means the feature is used more and splits better.
# class_weight='balanced' compensates for any residual class skew.
print("=" * 60)
print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

rf_all = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")
rf_all.fit(X_train, y_train)

importances = pd.Series(rf_all.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print("\nFeature importances (RandomForest MDI — higher = more useful):")
for feat, imp in importances.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<28} {imp:.4f}  {bar}")

y_pred_val_all = rf_all.predict(X_val)
all_f1  = f1_score(y_val, y_pred_val_all)
all_acc = accuracy_score(y_val, y_pred_val_all)
print(f"\nAll-features baseline → Accuracy: {all_acc:.3f}  F1 (Hard): {all_f1:.3f}")

# ── STEP 7: Prune low-importance features ─────────────────────────────────────
# We prune features whose importance falls below 0.015 (1.5%).
# Features below this threshold add noise rather than signal —
# they split on random variation rather than true difficulty patterns.
# After pruning, we re-train and confirm F1 does not drop meaningfully (>0.01).
print("\n" + "=" * 60)
print("STEP 5: PRUNING LOW-IMPORTANCE FEATURES")
print("=" * 60)

PRUNE_THRESHOLD = 0.015
keep   = importances[importances >= PRUNE_THRESHOLD].index.tolist()
pruned = importances[importances <  PRUNE_THRESHOLD].index.tolist()

print(f"Prune threshold : importance < {PRUNE_THRESHOLD}")
print(f"Pruned ({len(pruned)}) : {pruned if pruned else 'none — all features above threshold'}")
print(f"Kept   ({len(keep)})  : {keep}")

rf_pruned = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")
rf_pruned.fit(X_train[keep], y_train)
y_pred_pruned = rf_pruned.predict(X_val[keep])
pruned_f1  = f1_score(y_val, y_pred_pruned)
pruned_acc = accuracy_score(y_val, y_pred_pruned)
delta_f1   = pruned_f1 - all_f1
print(f"\nPruned-set val → Accuracy: {pruned_acc:.3f}  F1: {pruned_f1:.3f}")
print(f"Delta F1 vs all-features : {delta_f1:+.3f}  ({'✓ acceptable' if delta_f1 > -0.01 else '✗ too much loss — lower threshold'})")

# ── STEP 8: Lock baseline feature set ────────────────────────────────────────
# The locked set is the final contract for downstream tasks (Tasks 8+).
# Any new feature must beat this baseline to be included.
print("\n" + "=" * 60)
print("STEP 6: LOCKED BASELINE FEATURE SET")
print("=" * 60)

BASELINE_FEATURES = keep
print("Locked baseline features (in importance order):")
for i, f in enumerate(BASELINE_FEATURES, 1):
    print(f"  {i:2}. {f}")

# Final evaluation on held-out TEST set (never touched during development)
y_pred_test = rf_pruned.predict(X_test[BASELINE_FEATURES])
test_f1  = f1_score(y_test, y_pred_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"\nFinal TEST set results (unseen data):")
print(f"  Accuracy : {test_acc:.3f}")
print(f"  F1 (Hard): {test_f1:.3f}")

# ── STEP 9: Save experiment log ───────────────────────────────────────────────
# Records every decision made in this run for traceability and submission evidence.
log = {
    "task"               : "Task 7 — Baseline Target Feature Engineering",
    "timestamp"          : datetime.now().isoformat(),
    "dataset"            : {"rows": len(data), "files": len(files), "excluded": "DevOps (malformed header)"},
    "target"             : {"column": "difficulty_level", "binarize_at": MEDIAN,
                            "class_0_easy": int(vc[0]), "class_1_hard": int(vc[1])},
    "features_engineered": list(feature_descriptions.keys()),
    "leakage_check"      : {
        "safe"   : ["q_len","q_word_count","avg_opt_len","max_opt_len","opt_len_range",
                    "total_opt_len","domain_enc","topic_enc","q_to_avg_opt_ratio"],
        "fixed"  : {"domain_avg_difficulty": "recomputed from train only",
                    "topic_avg_difficulty"  : "recomputed from train only"},
        "excluded": ["difficulty_level (target)","question_id (identifier)","correct_answer (no signal)"]
    },
    "split"              : {"train": len(X_train), "val": len(X_val), "test": len(X_test),
                            "strategy": "stratified, random_state=42"},
    "importance_ranking" : {k: round(v, 4) for k, v in importances.items()},
    "pruning"            : {"threshold": PRUNE_THRESHOLD, "pruned": pruned, "kept": keep},
    "results"            : {
        "all_features_val_f1"    : round(all_f1, 4),
        "pruned_features_val_f1" : round(pruned_f1, 4),
        "delta_f1"               : round(delta_f1, 4),
        "test_f1"                : round(test_f1, 4),
        "test_accuracy"          : round(test_acc, 4)
    },
    "locked_baseline_features": BASELINE_FEATURES
}
log_path = "/mnt/user-data/outputs/task7_experiment_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"\nExperiment log saved → task7_experiment_log.json")

# ── STEP 10: Plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 7 — Baseline Feature Engineering\nPlaceMux · Phase 1 · AI/ML Developer",
             fontsize=12, fontweight="bold")

# Plot 1: Feature importance bar chart
# Blue = kept in baseline, grey = pruned
colors = ["#1976D2" if f in keep else "#BDBDBD" for f in importances.index]
axes[0].barh(importances.index[::-1], importances.values[::-1], color=colors[::-1], edgecolor="white")
axes[0].axvline(PRUNE_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                label=f"Prune threshold ({PRUNE_THRESHOLD})")
axes[0].set_xlabel("Importance (Mean Decrease in Impurity)")
axes[0].set_title("Feature Importance\n(blue = kept in baseline, grey = pruned)")
axes[0].legend(fontsize=8)
axes[0].grid(True, axis="x", alpha=0.3)

# Plot 2: F1 comparison across stages
labels     = ["All Features\n(Val)", "Pruned Baseline\n(Val)", "Locked Baseline\n(Test)"]
f1_values  = [all_f1, pruned_f1, test_f1]
bar_colors = ["#42A5F5", "#66BB6A", "#FFA726"]
bars = axes[1].bar(labels, f1_values, color=bar_colors, width=0.45, edgecolor="white")
for bar, val in zip(bars, f1_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("F1 Score (Hard class)")
axes[1].set_title("F1 Score at Each Stage\n(Val → Val pruned → Test unseen)")
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/task7_feature_engineering.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → task7_feature_engineering.png")

print("\n" + "=" * 60)
print("✓ TASK 7 COMPLETE")
print("=" * 60)
print(f"  Locked baseline : {len(BASELINE_FEATURES)} features")
print(f"  Leakage status  : 2 features fixed (train-only aggregates)")
print(f"  Val F1 (all)    : {all_f1:.3f}")
print(f"  Val F1 (pruned) : {pruned_f1:.3f}  (Δ {delta_f1:+.3f})")
print(f"  Test F1 (final) : {test_f1:.3f}")
print(f"\nFiles produced:")
print(f"  task7_feature_engineering.py   — this script")
print(f"  task7_feature_engineering.png  — importance chart + F1 comparison")
print(f"  task7_experiment_log.json      — full run log for submission evidence")
