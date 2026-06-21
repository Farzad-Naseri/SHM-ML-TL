import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==============================
# PATHS
# ==============================

data_path = r"E:\Farzad\AI for Damage Detection of Beams\ML\Hybrid\Data"
output_path = r"E:\Farzad\AI for Damage Detection of Beams\ML\Hybrid\Results\Model_Comparison"
os.makedirs(output_path, exist_ok=True)

# ==============================
# INPUT FILES
# ==============================

file_rms  = os.path.join(data_path, "Damage Detection (ΔRMS).xlsx")
file_peak = os.path.join(data_path, "Damage Detection (ΔPV).xlsx")
file_cf   = os.path.join(data_path, "Damage Detection (ΔCF).xlsx")
file_df   = os.path.join(data_path, "Damage Detection (Δf).xlsx")
file_mode = os.path.join(data_path, "Damage Detection (Deltamodes)TR.xlsx")

# ==============================
# READ DATA
# ==============================

def read_file(path):
    df = pd.read_excel(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

rms  = read_file(file_rms)
pv   = read_file(file_peak)
cf   = read_file(file_cf)
df_f = read_file(file_df)
mode = read_file(file_mode)

# target
y = pd.to_numeric(rms.iloc[:, 0], errors="coerce").fillna(0).astype(int)

# features
feature_blocks = [
    rms.iloc[:, 1:].add_prefix("RMS_"),
    pv.iloc[:, 1:].add_prefix("PV_"),
    cf.iloc[:, 1:].add_prefix("CF_"),
    df_f.iloc[:, 1:].add_prefix("DF_"),
    mode.iloc[:, 1:].add_prefix("MODE_")
]

X = pd.concat(feature_blocks, axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
feature_names = X.columns.tolist()

print("Feature shape:", X.shape)
print("Class counts:\n", y.value_counts())

# ==============================
# MODELS
# ==============================

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2))
    ]),

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, penalty="l2", solver="lbfgs"))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                                alpha=1e-4, learning_rate_init=1e-3, max_iter=2000,
                                early_stopping=True, random_state=42))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, subsample=0.8, random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                      class_weight="balanced", random_state=42))
    ]),

    "GaussianProcess": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessClassifier(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
            random_state=42
        ))
    ])
}

# ==============================
# HELPERS
# ==============================

def get_scores(estimator, X_test):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X_test)
    return estimator.predict(X_test)


def extract_feature_importance(estimator, X_ref, y_ref):
    model_step = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator

    if hasattr(model_step, "feature_importances_"):
        return np.asarray(model_step.feature_importances_, dtype=float), "Built-in"

    if hasattr(model_step, "coef_"):
        coef = np.asarray(model_step.coef_)
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            imp = np.mean(np.abs(coef), axis=0)
        return np.asarray(imp, dtype=float), "Absolute Coefficients"

    if hasattr(model_step, "coefs_") and len(model_step.coefs_) > 0:
        first_layer = np.asarray(model_step.coefs_[0])
        imp = np.mean(np.abs(first_layer), axis=1)
        return np.asarray(imp, dtype=float), "Mean Absolute First-Layer Weights"

    perm = permutation_importance(estimator, X_ref, y_ref, n_repeats=10, random_state=42, scoring="f1")
    return np.asarray(perm.importances_mean, dtype=float), "Permutation"


def save_top_importance_plot(df_imp, title, save_path, top_n=15):
    df_top = df_imp.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(df_top["Feature"], df_top["Importance"])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def permutation_leakage_test(estimator, X_data, y_data, cv, n_runs=10):
    rows = []
    rng = np.random.RandomState(42)

    for run in range(n_runs):
        y_perm = pd.Series(rng.permutation(y_data.values), index=y_data.index)
        y_true_all, y_pred_all, y_prob_all = [], [], []

        for train_idx, test_idx in cv.split(X_data, y_perm):
            est = clone(estimator)
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train, y_test = y_perm.iloc[train_idx], y_perm.iloc[test_idx]

            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            y_score = get_scores(est, X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(y_score)

        try:
            auc_val = roc_auc_score(y_true_all, y_prob_all)
        except Exception:
            auc_val = np.nan

        rows.append([
            run + 1,
            accuracy_score(y_true_all, y_pred_all),
            f1_score(y_true_all, y_pred_all, zero_division=0),
            auc_val
        ])

    return pd.DataFrame(rows, columns=["Run", "Accuracy", "F1", "ROC_AUC"])


def make_taylor_diagram(std_ref, model_stats, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    max_std = max([std_ref] + [v["std"] for v in model_stats.values()]) * 1.2
    rs = np.linspace(0, max_std, 6)
    ax.set_rgrids(rs[1:], angle=135)
    ax.set_ylim(0, max_std)

    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:.2f}" for c in corr_ticks])

    # reference point
    ax.plot(0, std_ref, "k*", markersize=14, label="Reference")

    # model points
    for name, vals in model_stats.items():
        theta = np.arccos(np.clip(vals["corr"], -1, 1))
        radius = vals["std"]
        ax.scatter(theta, radius, s=80, label=name)

    ax.set_title("Taylor Diagram", pad=25, fontsize=18)

    # move legend outside so it does not cover the diagram
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==============================
# CROSS VALIDATION + ALL OUTPUTS
# ==============================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
fold_rows = []
roc_data = {}
all_cm = {}
taylor_stats = {}

for name, base_model in models.items():
    print(f"\nTraining {name} ...")

    model_folder = os.path.join(output_path, name)
    os.makedirs(model_folder, exist_ok=True)

    y_true_oof = np.zeros(len(y), dtype=int)
    y_pred_oof = np.zeros(len(y), dtype=int)
    y_score_oof = np.zeros(len(y), dtype=float)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        est = clone(base_model)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        y_score = get_scores(est, X_test)

        y_true_oof[test_idx] = y_test.values
        y_pred_oof[test_idx] = y_pred
        y_score_oof[test_idx] = y_score

        fold_acc = accuracy_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test, y_score)
        except Exception:
            fold_auc = np.nan

        fold_metrics.append([fold, fold_acc, fold_f1, fold_auc])
        fold_rows.append([name, fold, fold_acc, fold_f1, fold_auc])

    # fold results
    fold_df = pd.DataFrame(fold_metrics, columns=["Fold", "Accuracy", "F1", "ROC_AUC"])
    fold_df.loc[len(fold_df)] = ["Mean", fold_df["Accuracy"].mean(), fold_df["F1"].mean(), fold_df["ROC_AUC"].mean()]
    fold_df.loc[len(fold_df)] = ["Std", fold_df.iloc[:5]["Accuracy"].std(), fold_df.iloc[:5]["F1"].std(), fold_df.iloc[:5]["ROC_AUC"].std()]
    fold_df.to_excel(os.path.join(model_folder, f"{name}_KFold_Results.xlsx"), index=False)

    # summary metrics
    acc = accuracy_score(y_true_oof, y_pred_oof)
    f1 = f1_score(y_true_oof, y_pred_oof, zero_division=0)
    auc = roc_auc_score(y_true_oof, y_score_oof)
    cm = confusion_matrix(y_true_oof, y_pred_oof)
    all_cm[name] = cm

    fpr, tpr, _ = roc_curve(y_true_oof, y_score_oof)
    roc_data[name] = (fpr, tpr, auc)

    # confusion matrix plot
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Damaged"], yticklabels=["Healthy", "Damaged"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Confusion_Matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ROC plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_ROC_Curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        clone(base_model), X, y, cv=skf, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    lc_df = pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_F1_Mean": train_mean,
        "Train_F1_Std": train_std,
        "Validation_F1_Mean": val_mean,
        "Validation_F1_Std": val_std
    })
    lc_df.to_excel(os.path.join(model_folder, f"{name}_Learning_Curve.xlsx"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training F1")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, marker="s", label="Validation F1")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training Samples")
    plt.ylabel("F1 Score")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Learning_Curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # fit full model for importance/permutation/taylor
    fitted_full = clone(base_model)
    fitted_full.fit(X, y)

    # feature importance
    importances, importance_method = extract_feature_importance(fitted_full, X, y)
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df.to_excel(os.path.join(model_folder, f"{name}_Feature_Importance.xlsx"), index=False)
    save_top_importance_plot(
        imp_df,
        f"Top Feature Importance - {name} ({importance_method})",
        os.path.join(model_folder, f"{name}_Top_Feature_Importance.png"),
        top_n=15
    )

    # permutation importance
    perm = permutation_importance(fitted_full, X, y, n_repeats=10, random_state=42, scoring="f1")
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std
    }).sort_values("Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    perm_df.to_excel(os.path.join(model_folder, f"{name}_Permutation_Importance.xlsx"), index=False)

    perm_plot_df = perm_df.rename(columns={"Permutation_Importance_Mean": "Importance"})[["Feature", "Importance"]]
    save_top_importance_plot(
        perm_plot_df,
        f"Top Permutation Importance - {name}",
        os.path.join(model_folder, f"{name}_Top_Permutation_Importance.png"),
        top_n=15
    )

    # leakage / shuffled-label test
    leakage_df = permutation_leakage_test(base_model, X, y, skf, n_runs=10)
    leakage_df.to_excel(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.xlsx"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(leakage_df["Run"], leakage_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(leakage_df["Run"], leakage_df["F1"], marker="s", label="F1")
    if leakage_df["ROC_AUC"].notna().any():
        plt.plot(leakage_df["Run"], leakage_df["ROC_AUC"], marker="^", label="ROC-AUC")
    plt.xlabel("Permutation Run")
    plt.ylabel("Score")
    plt.title(f"Permutation Leakage Test - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # final summary per model
    leakage_acc_mean = leakage_df["Accuracy"].mean()
    leakage_f1_mean = leakage_df["F1"].mean()
    leakage_auc_mean = leakage_df["ROC_AUC"].mean()

    final_summary_df = pd.DataFrame({
        "Metric": [
            "OOF Accuracy", "OOF F1", "OOF ROC_AUC",
            "CV Accuracy Mean", "CV Accuracy Std",
            "CV F1 Mean", "CV F1 Std",
            "CV ROC_AUC Mean", "CV ROC_AUC Std",
            "Permutation Accuracy Mean", "Permutation F1 Mean", "Permutation ROC_AUC Mean"
        ],
        "Value": [
            acc, f1, auc,
            fold_df.iloc[:5]["Accuracy"].mean(), fold_df.iloc[:5]["Accuracy"].std(),
            fold_df.iloc[:5]["F1"].mean(), fold_df.iloc[:5]["F1"].std(),
            fold_df.iloc[:5]["ROC_AUC"].mean(), fold_df.iloc[:5]["ROC_AUC"].std(),
            leakage_acc_mean, leakage_f1_mean, leakage_auc_mean
        ]
    })
    final_summary_df.to_excel(os.path.join(model_folder, f"{name}_Final_Summary.xlsx"), index=False)

    summary_rows.append([
        name,
        acc,
        f1,
        auc,
        fold_df.iloc[:5]["Accuracy"].mean(),
        fold_df.iloc[:5]["Accuracy"].std(),
        fold_df.iloc[:5]["F1"].mean(),
        fold_df.iloc[:5]["F1"].std(),
        fold_df.iloc[:5]["ROC_AUC"].mean(),
        fold_df.iloc[:5]["ROC_AUC"].std(),
        leakage_acc_mean,
        leakage_f1_mean,
        leakage_auc_mean,
        importance_method
    ])

    # taylor stats from OOF scores
    y_ref = y_true_oof.astype(float)
    y_model = y_score_oof.astype(float)
    corr_val = np.corrcoef(y_ref, y_model)[0, 1]
    if np.isnan(corr_val):
        corr_val = 0.0
    taylor_stats[name] = {
        "std": float(np.std(y_model, ddof=1)),
        "corr": float(np.clip(corr_val, -1, 1))
    }

# ==============================
# MASTER TABLES
# ==============================

results_df = pd.DataFrame(summary_rows, columns=[
    "Model", "Accuracy", "F1", "ROC_AUC",
    "CV_Accuracy_Mean", "CV_Accuracy_Std",
    "CV_F1_Mean", "CV_F1_Std",
    "CV_ROC_AUC_Mean", "CV_ROC_AUC_Std",
    "Permutation_Accuracy_Mean", "Permutation_F1_Mean", "Permutation_ROC_AUC_Mean",
    "Feature_Importance_Method"
]).sort_values("F1", ascending=False).reset_index(drop=True)
results_df.to_excel(os.path.join(output_path, "Model_Comparison.xlsx"), index=False)

fold_compare_df = pd.DataFrame(fold_rows, columns=["Model", "Fold", "Accuracy", "F1", "ROC_AUC"])
fold_compare_df.to_excel(os.path.join(output_path, "All_Models_Foldwise_Results.xlsx"), index=False)

print("\nOverall summary:")
print(results_df)

# ==============================
# COMPARISON BAR PLOTS
# ==============================

for metric in ["Accuracy", "F1", "ROC_AUC"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = results_df.sort_values(metric, ascending=False)
    sns.barplot(data=plot_df, x="Model", y=metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

# fold-wise comparison plots
for metric in ["Accuracy", "F1", "ROC_AUC"]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fold_compare_df, x="Model", y=metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Fold-wise {metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"Foldwise_{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

# comparison heatmap
heatmap_df = results_df.set_index("Model")[["Accuracy", "F1", "ROC_AUC"]]
plt.figure(figsize=(8, 5.5))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Model Performance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Model_Performance_Heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# radar chart
radar_metrics = ["Accuracy", "F1", "ROC_AUC"]
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for _, row in results_df.iterrows():
    values = row[radar_metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1.5, label=row["Model"])
    ax.fill(angles, values, alpha=0.05)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_title("Radar Comparison of Models")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Radar_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==============================
# ROC CURVES COMPARISON
# ==============================

plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "ROC_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==============================
# CONFUSION MATRIX COMPARISON
# ==============================

n_models = len(models)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, (name, cm) in zip(axes, all_cm.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Healthy", "Damaged"], yticklabels=["Healthy", "Damaged"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(output_path, "All_Models_Confusion_Matrices.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==============================
# LEARNING CURVE COMPARISON
# ==============================

plt.figure(figsize=(9, 6))
for name, base_model in models.items():
    train_sizes, _, val_scores = learning_curve(
        clone(base_model), X, y, cv=skf, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label=name)
plt.xlabel("Training Samples")
plt.ylabel("Validation F1 Score")
plt.title("Learning Curve Comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Learning_Curve_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# ==============================
# TAYLOR DIAGRAM
# ==============================

std_ref = float(np.std(y.astype(float), ddof=1))
make_taylor_diagram(std_ref, taylor_stats, os.path.join(output_path, "Taylor_Diagram.png"))

print("\nAll model comparison results and graphs have been saved.")
