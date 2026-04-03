#!/usr/bin/env python3
"""
Analysis 1: Genome-Based Prediction of Plasticity Benefit

Tests whether genome/topology features can predict which networks benefit
from plasticity. Uses logistic regression and random forest with 5-fold
stratified CV on B0.5+ CartPole, plus cross-task transfer to Acrobot.

Figures:
  F0_feature_importance   — RF feature importance + permutation importance
  F1_roc_curves           — ROC for "improved" classification
  F2_regression_scatter   — Predicted vs actual best_delta_reward
  F3_cross_task_transfer  — CartPole model evaluated on Acrobot
  F4_eta_sign_prediction  — Confusion matrix + importance for optimal η sign

Usage:
  python scripts/analyze_genome_prediction.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from MorphoNAS_PL.analysis_utils import (
    STRATA_ORDER,
    STRATUM_COLORS,
    STRATUM_LABELS,
    apply_publication_style,
    load_sweep,
    save_figure,
    strata_present_in,
)
from MorphoNAS_PL.genome_features import (
    FEATURE_COLUMNS,
    load_genome_features,
)

logger = logging.getLogger(__name__)
apply_publication_style()

# ── Paths ────────────────────────────────────────────────────────────
B05P_SWEEP_DIR = "experiments/B0.5+/sweep"
B05P_POOL_DIR = "experiments/B0.5+/pool_subsample"
ACROBOT_SWEEP_DIR = "experiments/acrobot/sweep_static"
ACROBOT_NS_SWEEP_DIR = "experiments/acrobot/sweep_nonstationary/sweep"
ACROBOT_POOL_DIR = "experiments/acrobot/pool"
DEFAULT_OUTPUT_DIR = "experiments/B0.5+/analysis/prediction"

NON_WEAK_STRATA = ["low_mid", "high_mid", "near_perfect", "perfect"]


# ── Data preparation ────────────────────────────────────────────────

def prepare_per_network_targets(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-network targets from sweep data.

    Returns DataFrame with columns:
        network_id, stratum, improved (oracle), best_delta_reward,
        best_eta, best_decay, optimal_eta_sign
    """
    # Per-network oracle: best delta_reward across all (eta, decay) pairs
    idx = sweep_df.groupby("network_id")["delta_reward"].idxmax()
    oracle = sweep_df.loc[idx, [
        "network_id", "stratum", "delta_reward", "eta", "decay",
    ]].copy()
    oracle = oracle.rename(columns={
        "delta_reward": "best_delta_reward",
        "eta": "best_eta",
        "decay": "best_decay",
    })
    oracle["improved"] = (oracle["best_delta_reward"] > 0).astype(int)
    oracle["optimal_eta_sign"] = np.sign(oracle["best_eta"]).astype(int)
    # Classify: -1 = anti-Hebbian, 0 = no plasticity, 1 = Hebbian
    return oracle.reset_index(drop=True)


def build_feature_matrix(
    genome_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge features with targets, return (X, merged_df)."""
    merged = targets_df.merge(genome_df, on="network_id", suffixes=("", "_genome"))
    # Use stratum from targets (sweep-derived) if available
    if "stratum_genome" in merged.columns:
        merged = merged.drop(columns=["stratum_genome"])

    # Filter to non-Weak
    merged = merged[merged["stratum"].isin(NON_WEAK_STRATA)].copy()

    X = merged[feature_cols].values.astype(np.float64)
    # Replace NaN with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            col[mask] = median_val

    return X, merged


# ── Figures ──────────────────────────────────────────────────────────

def figure_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    fig_dir: str,
) -> dict:
    """F0: RF feature importance + permutation importance."""
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    # Gini importance
    gini_imp = rf.feature_importances_
    sorted_idx = np.argsort(gini_imp)[::-1][:15]

    # Permutation importance
    perm_result = permutation_importance(
        rf, X, y, n_repeats=10, random_state=42, n_jobs=-1,
    )
    perm_imp = perm_result.importances_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gini
    ax = axes[0]
    names = [feature_names[i] for i in sorted_idx]
    vals = gini_imp[sorted_idx]
    ax.barh(range(len(names)), vals, color="#2196F3")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Gini Importance")
    ax.set_title("(a) Gini Feature Importance")
    ax.invert_yaxis()

    # Permutation
    ax = axes[1]
    sorted_perm_idx = np.argsort(perm_imp)[::-1][:15]
    names_p = [feature_names[i] for i in sorted_perm_idx]
    vals_p = perm_imp[sorted_perm_idx]
    ax.barh(range(len(names_p)), vals_p, color="#FF9800")
    ax.set_yticks(range(len(names_p)))
    ax.set_yticklabels(names_p, fontsize=9)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("(b) Permutation Feature Importance")
    ax.invert_yaxis()

    fig.suptitle("Genome Feature Importance for Predicting Plasticity Benefit")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F0_feature_importance")

    return {
        "top_gini": {feature_names[i]: float(gini_imp[i]) for i in sorted_idx},
        "top_perm": {feature_names[i]: float(perm_imp[i]) for i in sorted_perm_idx},
    }


def figure_roc_curves(
    X: np.ndarray,
    y: np.ndarray,
    fig_dir: str,
) -> dict:
    """F1: ROC curves for 'improved' classification (5-fold CV)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"Logistic Regression": "#2196F3", "Random Forest": "#4CAF50"}

    for name, model in models.items():
        y_prob = cross_val_predict(
            model, X_scaled, y, cv=cv, method="predict_proba",
        )[:, 1]
        auc = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax.plot(fpr, tpr, color=colors[name], label=f"{name} (AUC={auc:.3f})", lw=2)
        results[name] = {"auc": float(auc)}

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Predicting Which Networks Benefit from Plasticity")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F1_roc_curves")

    return results


def figure_regression_scatter(
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    fig_dir: str,
) -> dict:
    """F2: Predicted vs actual best_delta_reward."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1,
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results = {}

    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for idx, (name, model) in enumerate(models.items()):
        y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        results[name] = {"r2": float(r2), "rmse": rmse}

        ax = axes[idx]
        for s in NON_WEAK_STRATA:
            mask = strata == s
            if mask.any():
                ax.scatter(
                    y[mask], y_pred[mask],
                    alpha=0.3, s=10,
                    color=STRATUM_COLORS.get(s, "#999"),
                    label=STRATUM_LABELS.get(s, s),
                )
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Actual best Δreward")
        ax.set_ylabel("Predicted best Δreward")
        ax.set_title(f"({chr(97+idx)}) {name} (R²={r2:.3f}, RMSE={rmse:.1f})")
        ax.legend(fontsize=8, markerscale=3)

    fig.suptitle("Regression: Predicting Magnitude of Plasticity Benefit")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F2_regression_scatter")

    return results


def figure_cross_task_transfer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_strata: np.ndarray,
    fig_dir: str,
) -> dict:
    """F3: CartPole-trained model evaluated on Acrobot."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # Classification: improved
    rf_clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_clf.fit(X_train_s, y_train)
    y_pred_prob = rf_clf.predict_proba(X_test_s)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC on Acrobot
    ax = axes[0]
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        ax.plot(fpr, tpr, color="#4CAF50", lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        results["transfer_auc"] = float(auc)
    except ValueError:
        ax.text(0.5, 0.5, "Insufficient class variation", ha="center", va="center")
        results["transfer_auc"] = None
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) Cross-Task Transfer ROC (Train: CartPole, Test: Acrobot)")
    ax.legend(loc="lower right")

    # Accuracy per stratum
    ax = axes[1]
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    strata_accs = {}
    for s in NON_WEAK_STRATA:
        mask = test_strata == s
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred_binary[mask])
            strata_accs[s] = float(acc)

    if strata_accs:
        bars = ax.bar(
            range(len(strata_accs)),
            list(strata_accs.values()),
            color=[STRATUM_COLORS.get(s, "#999") for s in strata_accs],
        )
        ax.set_xticks(range(len(strata_accs)))
        ax.set_xticklabels(
            [STRATUM_LABELS.get(s, s) for s in strata_accs], rotation=30, ha="right",
        )
        ax.set_ylabel("Accuracy")
        ax.set_title("(b) Per-Stratum Transfer Accuracy")
        ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.5)
    results["per_stratum_accuracy"] = strata_accs

    fig.suptitle("Cross-Task Transfer: CartPole Model on Acrobot")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F3_cross_task_transfer")

    return results


def figure_eta_sign_prediction(
    X: np.ndarray,
    y_sign: np.ndarray,
    feature_names: list[str],
    fig_dir: str,
) -> dict:
    """F4: Optimal η sign prediction."""
    # Filter to anti-Hebbian vs Hebbian only (exclude η=0)
    mask = y_sign != 0
    X_f = X[mask]
    y_f = (y_sign[mask] == -1).astype(int)  # 1 = anti-Hebbian

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_f)

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf, X_scaled, y_f, cv=cv)

    # Feature importance
    rf.fit(X_scaled, y_f)
    imp = rf.feature_importances_
    sorted_idx = np.argsort(imp)[::-1][:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_f, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Hebbian", "Anti-Hebbian"])
    ax.set_yticklabels(["Hebbian", "Anti-Hebbian"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    acc = accuracy_score(y_f, y_pred)
    ax.set_title(f"(a) Confusion Matrix (Acc={acc:.3f})")

    # Feature importance
    ax = axes[1]
    names = [feature_names[i] for i in sorted_idx]
    vals = imp[sorted_idx]
    ax.barh(range(len(names)), vals, color="#9C27B0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("(b) Features Predicting Optimal η Sign")
    ax.invert_yaxis()

    fig.suptitle("Predicting Anti-Hebbian vs Hebbian Preference")
    fig.tight_layout()
    save_figure(fig, fig_dir, "F4_eta_sign_prediction")

    return {
        "accuracy": float(acc),
        "n_anti_hebbian": int(y_f.sum()),
        "n_hebbian": int((~y_f.astype(bool)).sum()),
        "top_features": {feature_names[i]: float(imp[i]) for i in sorted_idx},
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analysis 1: Genome-Based Prediction of Plasticity Benefit",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    fig_dir = os.path.join(args.output_dir, "figures")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading B0.5+ sweep data...")
    sweep_df = load_sweep(B05P_SWEEP_DIR)
    sweep_df = sweep_df[sweep_df["stratum"].isin(NON_WEAK_STRATA)]

    print("Loading B0.5+ genome features...")
    genome_df = load_genome_features(B05P_POOL_DIR)

    print("Computing per-network targets...")
    targets_df = prepare_per_network_targets(sweep_df)

    print("Building feature matrix...")
    X, merged = build_feature_matrix(genome_df, targets_df, FEATURE_COLUMNS)

    y_improved = merged["improved"].values
    y_delta = merged["best_delta_reward"].values
    y_sign = merged["optimal_eta_sign"].values
    strata = merged["stratum"].values

    print(f"  Feature matrix: {X.shape[0]} networks x {X.shape[1]} features")
    print(f"  Improved: {y_improved.sum()}/{len(y_improved)} "
          f"({100*y_improved.mean():.1f}%)")

    # ── CartPole analyses ────────────────────────────────────────────
    print("\n--- F0: Feature Importance ---")
    imp_results = figure_feature_importance(X, y_improved, FEATURE_COLUMNS, fig_dir)

    print("--- F1: ROC Curves ---")
    roc_results = figure_roc_curves(X, y_improved, fig_dir)

    print("--- F2: Regression ---")
    reg_results = figure_regression_scatter(X, y_delta, strata, fig_dir)

    print("--- F4: η Sign Prediction ---")
    eta_results = figure_eta_sign_prediction(X, y_sign, FEATURE_COLUMNS, fig_dir)

    # ── Cross-task transfer ──────────────────────────────────────────
    transfer_results = {}
    try:
        print("\nLoading Acrobot data for cross-task transfer...")
        acrobot_sweep_file = os.path.join(
            ACROBOT_SWEEP_DIR, "acrobot_static_sweep.parquet",
        )
        acrobot_sweep = pd.read_parquet(acrobot_sweep_file)
        acrobot_sweep = acrobot_sweep[
            acrobot_sweep["stratum"].isin(NON_WEAK_STRATA)
        ]
        acrobot_genome = load_genome_features(ACROBOT_POOL_DIR)
        acrobot_targets = prepare_per_network_targets(acrobot_sweep)

        X_acrobot, merged_acrobot = build_feature_matrix(
            acrobot_genome, acrobot_targets, FEATURE_COLUMNS,
        )
        y_acrobot = merged_acrobot["improved"].values
        strata_acrobot = merged_acrobot["stratum"].values

        print(f"  Acrobot: {X_acrobot.shape[0]} networks")

        print("--- F3: Cross-Task Transfer ---")
        transfer_results = figure_cross_task_transfer(
            X, y_improved, X_acrobot, y_acrobot, strata_acrobot, fig_dir,
        )
    except Exception as e:
        logger.warning("Could not run cross-task transfer: %s", e)

    # ── Save summary ─────────────────────────────────────────────────
    summary = {
        "n_cartpole_networks": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "improvement_rate": float(y_improved.mean()),
        "feature_importance": imp_results,
        "roc": roc_results,
        "regression": reg_results,
        "eta_sign": eta_results,
        "cross_task_transfer": transfer_results,
    }
    summary_path = os.path.join(args.output_dir, "prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print text summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENOME-BASED PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Networks: {X.shape[0]} (non-Weak B0.5+)")
    print(f"  Features: {X.shape[1]}")
    print(f"  Improvement rate: {100*y_improved.mean():.1f}%")
    for name, r in roc_results.items():
        print(f"  {name} AUC: {r['auc']:.3f}")
    for name, r in reg_results.items():
        print(f"  {name} R²={r['r2']:.3f}, RMSE={r['rmse']:.1f}")
    print(f"  η sign accuracy: {eta_results['accuracy']:.3f}")
    if transfer_results.get("transfer_auc") is not None:
        print(f"  Cross-task AUC: {transfer_results['transfer_auc']:.3f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
