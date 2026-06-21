import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==============================
# PATHS
# ==============================

data_path = r"E:\Farzad\AI for Damage Detection of Beams\ML\Hybrid\Data Cal"
output_path = r"E:\Farzad\AI for Damage Detection of Beams\ML\Hybrid\Results\3. Severity_Model_Comparison1"
os.makedirs(output_path, exist_ok=True)

# ==============================
# INPUT FILES
# ==============================

file_rms  = os.path.join(data_path, "Damage Severity (ΔRMS).xlsx")
file_peak = os.path.join(data_path, "Damage Severity (ΔPV).xlsx")
file_cf   = os.path.join(data_path, "Damage Severity (ΔCF).xlsx")
file_df   = os.path.join(data_path, "Damage Severity (Δf).xlsx")
file_mode = os.path.join(data_path, "Damage Severity (Deltamodes)TR.xlsx")

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

y = pd.to_numeric(rms.iloc[:, 0], errors="coerce").fillna(0.0)

feature_blocks = [
    rms.iloc[:, 1:].add_prefix("RMS_"),
    pv.iloc[:, 1:].add_prefix("PV_"),
    cf.iloc[:, 1:].add_prefix("CF_"),
    df_f.iloc[:, 1:].add_prefix("DF_"),
    mode.iloc[:, 1:].add_prefix("MODE_")
]

X = pd.concat(feature_blocks, axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X.columns = [f"F{i+1}_{col}" for i, col in enumerate(X.columns)]
feature_names = X.columns.tolist()

valid_mask = y.notna()
X = X.loc[valid_mask].reset_index(drop=True)
y = y.loc[valid_mask].reset_index(drop=True)

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Severity target summary:")
print(y.describe())

# ==============================
# MODELS
# ==============================

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=16,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        ))
    ]),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=10,
            gamma=0.1,
            epsilon=0.5
        ))
    ]),

    "GaussianProcess": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        ))
    ])
}

# ==============================
# HELPERS
# ==============================

def add_bar_labels(ax, fmt="{:.3f}", fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 4),
            textcoords="offset points"
        )


def extract_feature_importance(estimator, X_ref, y_ref, scoring_metric):
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

    perm = permutation_importance(
        estimator, X_ref, y_ref,
        n_repeats=10, random_state=42, scoring=scoring_metric, n_jobs=1
    )
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

    for run in range(n_runs):
        y_perm = y_data.sample(frac=1, random_state=42 + run).reset_index(drop=True)
        pred_all = np.zeros(len(y_perm), dtype=float)

        for train_idx, test_idx in cv.split(X_data):
            est = clone(estimator)
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            y_train = y_perm.iloc[train_idx]

            est.fit(X_train, y_train)
            pred_all[test_idx] = est.predict(X_test)

        rows.append([
            run + 1,
            mean_absolute_error(y_perm, pred_all),
            np.sqrt(mean_squared_error(y_perm, pred_all)),
            r2_score(y_perm, pred_all)
        ])

    return pd.DataFrame(rows, columns=["Run", "MAE", "RMSE", "R2"])


def make_taylor_diagram(std_ref, model_stats, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    max_std = max([std_ref] + [v["std"] for v in model_stats.values()]) * 1.2
    rs = np.linspace(0, max_std, 6)
    ax.set_rgrids(rs[1:], angle=135, labels=[f"{r:.2f}" for r in rs[1:]])
    ax.set_ylim(0, max_std)

    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    ax.set_thetagrids(
        np.degrees(np.arccos(corr_ticks)),
        labels=[f"{c:.2f}" for c in corr_ticks]
    )

    ax.plot(0, std_ref, "k*", markersize=14, label="Reference")

    for name, vals in model_stats.items():
        theta = np.arccos(np.clip(vals["corr"], -1, 1))
        radius = vals["std"]
        ax.scatter(theta, radius, s=80, label=name)

    ax.set_title("Taylor Diagram", pad=25, fontsize=18)
    fig.text(0.80, 0.08, "Radial axis: Standard Deviation (severity units)", ha="center", fontsize=10)
    fig.text(0.22, 0.92, "Angular axis: Correlation Coefficient (unitless)", ha="center", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==============================
# CROSS VALIDATION + ALL OUTPUTS
# ==============================

cv = KFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
fold_rows = []
all_pred_true = {}
taylor_stats = {}
leakage_summary_rows = []
learning_curve_summary = []

for name, base_model in models.items():
    print(f"\nTraining {name} ...")

    model_folder = os.path.join(output_path, name)
    os.makedirs(model_folder, exist_ok=True)

    oof_pred = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        est = clone(base_model)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        oof_pred[test_idx] = pred

        fold_mae = mean_absolute_error(y_test, pred)
        fold_rmse = np.sqrt(mean_squared_error(y_test, pred))
        fold_r2 = r2_score(y_test, pred)

        fold_metrics.append([fold, fold_mae, fold_rmse, fold_r2])
        fold_rows.append([name, fold, fold_mae, fold_rmse, fold_r2])

    fold_df = pd.DataFrame(fold_metrics, columns=["Fold", "MAE", "RMSE", "R2"])
    fold_df.loc[len(fold_df)] = [
        "Mean",
        fold_df["MAE"].mean(),
        fold_df["RMSE"].mean(),
        fold_df["R2"].mean()
    ]
    fold_df.loc[len(fold_df)] = [
        "Std",
        fold_df.iloc[:5]["MAE"].std(ddof=1),
        fold_df.iloc[:5]["RMSE"].std(ddof=1),
        fold_df.iloc[:5]["R2"].std(ddof=1)
    ]
    fold_df.to_excel(os.path.join(model_folder, f"{name}_KFold_Results.xlsx"), index=False)

    mae = mean_absolute_error(y, oof_pred)
    rmse = np.sqrt(mean_squared_error(y, oof_pred))
    r2 = r2_score(y, oof_pred)
    all_pred_true[name] = (y.values.copy(), oof_pred.copy())

    # predicted vs true with ±30% boundaries
    plt.figure(figsize=(7, 7))
    plt.scatter(y.values, oof_pred, alpha=0.7)
    min_val = min(y.min(), oof_pred.min())
    max_val = max(y.max(), oof_pred.max())
    x_line = np.linspace(min_val, max_val, 300)

    plt.plot(x_line, x_line, "k--", linewidth=2, label="Perfect Prediction")
    plt.plot(x_line, 1.3 * x_line, "r--", linewidth=1.8, label="+30% Error Boundary")
    plt.plot(x_line, 0.7 * x_line, "r--", linewidth=1.8, label="-30% Error Boundary")

    plt.xlabel("True Severity")
    plt.ylabel("Predicted Severity")
    plt.title(f"Predicted vs True Severity (Out-of-Fold) - {name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Predicted_vs_True.png"), dpi=300, bbox_inches="tight")
    plt.close()

    abs_error = np.abs(y.values - oof_pred)
    error_df = pd.DataFrame({
        "True_Severity": y.values,
        "Predicted_Severity": oof_pred,
        "Absolute_Error": abs_error
    })
    error_df.to_excel(os.path.join(model_folder, f"{name}_Predictions_and_Errors.xlsx"), index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(abs_error, bins=20)
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title(f"Severity Error Histogram - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Error_Histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(abs_error)), abs_error, marker="o", linestyle="-")
    plt.xlabel("Sample Index")
    plt.ylabel("Absolute Error")
    plt.title(f"Severity Error by Sample - {name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Error_By_Sample.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fold_only = fold_df.iloc[:5].copy()
    x = np.arange(1, 6)
    plt.figure(figsize=(8, 5))
    plt.plot(x, fold_only["MAE"], marker="o", label="MAE")
    plt.plot(x, fold_only["RMSE"], marker="o", label="RMSE")
    plt.plot(x, fold_only["R2"], marker="o", label="R2")
    plt.xlabel("Fold")
    plt.ylabel("Metric Value")
    plt.title(f"Cross-Validation Fold Performance - {name}")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Fold_Performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # learning curve: use R2 for all models
    train_sizes, train_scores, val_scores = learning_curve(
        clone(base_model), X, y, cv=cv, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    lc_df = pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_R2_Mean": train_mean,
        "Train_R2_Std": train_std,
        "Validation_R2_Mean": val_mean,
        "Validation_R2_Std": val_std
    })
    lc_df.to_excel(os.path.join(model_folder, f"{name}_Learning_Curve.xlsx"), index=False)

    learning_curve_summary.append([
        name,
        train_mean[-1],
        val_mean[-1]
    ])

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, marker="s", label="Validation")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Training Samples")
    plt.ylabel("R2")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Learning_Curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fitted_full = clone(base_model)
    fitted_full.fit(X, y)

    importance_scoring = "r2"

    importances, importance_method = extract_feature_importance(fitted_full, X, y, importance_scoring)
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df.to_excel(os.path.join(model_folder, f"{name}_Feature_Importance.xlsx"), index=False)
    save_top_importance_plot(
        imp_df,
        f"Top Feature Importance - {name} ({importance_method})",
        os.path.join(model_folder, f"{name}_Top_Feature_Importance.png"),
        top_n=15
    )

    perm = permutation_importance(
        fitted_full, X, y, n_repeats=10, random_state=42,
        scoring=importance_scoring, n_jobs=1
    )
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

    # leakage test per model
    leakage_df = permutation_leakage_test(base_model, X, y, cv, n_runs=10)
    leakage_df.to_excel(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.xlsx"), index=False)

    leakage_summary_rows.append([
        name,
        leakage_df["MAE"].mean(),
        leakage_df["RMSE"].mean(),
        leakage_df["R2"].mean()
    ])

    plt.figure(figsize=(7, 5))
    plt.plot(leakage_df["Run"], leakage_df["MAE"], marker="o", label="MAE")
    plt.plot(leakage_df["Run"], leakage_df["RMSE"], marker="s", label="RMSE")
    plt.plot(leakage_df["Run"], leakage_df["R2"], marker="^", label="R2")
    plt.xlabel("Permutation Run")
    plt.ylabel("Metric Value")
    plt.title(f"Permutation Leakage Test - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    final_summary_df = pd.DataFrame({
        "Metric": [
            "OOF MAE", "OOF RMSE", "OOF R2",
            "CV MAE Mean", "CV MAE Std",
            "CV RMSE Mean", "CV RMSE Std",
            "CV R2 Mean", "CV R2 Std",
            "Permutation MAE Mean", "Permutation RMSE Mean", "Permutation R2 Mean"
        ],
        "Value": [
            mae, rmse, r2,
            fold_df.iloc[:5]["MAE"].mean(), fold_df.iloc[:5]["MAE"].std(ddof=1),
            fold_df.iloc[:5]["RMSE"].mean(), fold_df.iloc[:5]["RMSE"].std(ddof=1),
            fold_df.iloc[:5]["R2"].mean(), fold_df.iloc[:5]["R2"].std(ddof=1),
            leakage_df["MAE"].mean(), leakage_df["RMSE"].mean(), leakage_df["R2"].mean()
        ]
    })
    final_summary_df.to_excel(os.path.join(model_folder, f"{name}_Final_Summary.xlsx"), index=False)

    summary_rows.append([
        name,
        mae,
        rmse,
        r2,
        fold_df.iloc[:5]["MAE"].mean(),
        fold_df.iloc[:5]["MAE"].std(ddof=1),
        fold_df.iloc[:5]["RMSE"].mean(),
        fold_df.iloc[:5]["RMSE"].std(ddof=1),
        fold_df.iloc[:5]["R2"].mean(),
        fold_df.iloc[:5]["R2"].std(ddof=1),
        leakage_df["MAE"].mean(),
        leakage_df["RMSE"].mean(),
        leakage_df["R2"].mean(),
        importance_method
    ])

    corr_val = np.corrcoef(y.values.astype(float), oof_pred.astype(float))[0, 1]
    if np.isnan(corr_val):
        corr_val = 0.0
    taylor_stats[name] = {
        "std": float(np.std(oof_pred.astype(float), ddof=1)),
        "corr": float(np.clip(corr_val, -1, 1))
    }

# ==============================
# MASTER TABLES
# ==============================

results_df = pd.DataFrame(summary_rows, columns=[
    "Model", "MAE", "RMSE", "R2",
    "CV_MAE_Mean", "CV_MAE_Std",
    "CV_RMSE_Mean", "CV_RMSE_Std",
    "CV_R2_Mean", "CV_R2_Std",
    "Permutation_MAE_Mean", "Permutation_RMSE_Mean", "Permutation_R2_Mean",
    "Feature_Importance_Method"
]).sort_values(["R2", "MAE"], ascending=[False, True]).reset_index(drop=True)
results_df.to_excel(os.path.join(output_path, "Severity_Model_Comparison.xlsx"), index=False)

fold_compare_df = pd.DataFrame(fold_rows, columns=["Model", "Fold", "MAE", "RMSE", "R2"])
fold_compare_df.to_excel(os.path.join(output_path, "Severity_All_Models_Foldwise_Results.xlsx"), index=False)

leakage_compare_df = pd.DataFrame(
    leakage_summary_rows,
    columns=["Model", "Leakage_MAE", "Leakage_RMSE", "Leakage_R2"]
)
leakage_compare_df.to_excel(os.path.join(output_path, "Severity_Leakage_Comparison.xlsx"), index=False)

print("\nOverall summary:")
print(results_df)

# ==============================
# PERMUTATION DROP COMPARISON
# ==============================

perm_compare_df = results_df[[
    "Model", "MAE", "RMSE", "R2",
    "Permutation_MAE_Mean", "Permutation_RMSE_Mean", "Permutation_R2_Mean"
]].copy()

perm_compare_df["MAE_Increase"] = perm_compare_df["Permutation_MAE_Mean"] - perm_compare_df["MAE"]
perm_compare_df["RMSE_Increase"] = perm_compare_df["Permutation_RMSE_Mean"] - perm_compare_df["RMSE"]
perm_compare_df["R2_Drop"] = perm_compare_df["R2"] - perm_compare_df["Permutation_R2_Mean"]

perm_compare_df.to_excel(os.path.join(output_path, "Severity_Permutation_Drop_Comparison.xlsx"), index=False)

# ==============================
# COMPARISON BAR PLOTS
# ==============================

for metric in ["R2"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = results_df.sort_values(metric, ascending=False)
    ax = sns.barplot(data=plot_df, x="Model", y=metric)
    add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

for metric in ["MAE", "RMSE"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = results_df.sort_values(metric, ascending=True)
    ax = sns.barplot(data=plot_df, x="Model", y=metric)
    add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

for metric in ["MAE_Increase", "RMSE_Increase"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = perm_compare_df.sort_values(metric, ascending=False)
    ax = sns.barplot(data=plot_df, x="Model", y=metric)
    add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Error Increase")
    plt.title(f"{metric.replace('_', ' ')} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(10, 5.5))
plot_df = perm_compare_df.sort_values("R2_Drop", ascending=False)
ax = sns.barplot(data=plot_df, x="Model", y="R2_Drop")
add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.ylabel("R2 Drop")
plt.title("R2 Drop Comparison")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "R2_Drop_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

for metric in ["MAE", "RMSE", "R2"]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fold_compare_df, x="Model", y=metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Fold-wise {metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"Foldwise_{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

heatmap_df = results_df.set_index("Model")[["MAE", "RMSE", "R2"]]
plt.figure(figsize=(8, 5.5))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Severity Model Performance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_Model_Performance_Heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# radar chart
radar_metrics = ["MAE", "RMSE", "R2"]
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
ax.set_title("Radar Comparison of Severity Models")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_Radar_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# learning curve comparison
plt.figure(figsize=(9, 6))
for name, base_model in models.items():
    train_sizes, _, val_scores = learning_curve(
        clone(base_model), X, y, cv=cv, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1
    )
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label=name)
plt.xlabel("Training Samples")
plt.ylabel("Validation R2")
plt.title("Learning Curve Comparison - Severity Models")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_Learning_Curve_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# training vs validation R2 comparison
lc_compare_df = pd.DataFrame(
    learning_curve_summary,
    columns=["Model", "Training_R2_Final", "Validation_R2_Final"]
)
lc_compare_df.to_excel(os.path.join(output_path, "Severity_Training_Validation_R2_Comparison.xlsx"), index=False)

plot_df = lc_compare_df.melt(
    id_vars="Model",
    value_vars=["Training_R2_Final", "Validation_R2_Final"],
    var_name="Type",
    value_name="R2"
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=plot_df, x="Model", y="R2", hue="Type")
add_bar_labels(ax, fmt="{:.3f}", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.title("Training vs Validation R2 Comparison")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_Training_vs_Validation_R2_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# predicted vs true comparison with ±30% boundaries
n_models = len(models)
n_cols = 4
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()
for ax, (name, (y_true_vals, y_pred_vals)) in zip(axes, all_pred_true.items()):
    ax.scatter(y_true_vals, y_pred_vals, alpha=0.6)
    min_val = min(np.min(y_true_vals), np.min(y_pred_vals))
    max_val = max(np.max(y_true_vals), np.max(y_pred_vals))
    x_line = np.linspace(min_val, max_val, 300)
    ax.plot(x_line, x_line, "k--", linewidth=1.6)
    ax.plot(x_line, 1.3 * x_line, "r--", linewidth=1.2)
    ax.plot(x_line, 0.7 * x_line, "r--", linewidth=1.2)
    ax.set_title(name)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
for ax in axes[len(all_pred_true):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_All_Models_Predicted_vs_True.png"), dpi=300, bbox_inches="tight")
plt.close()

# leakage comparison graphs across all models
for metric in ["Leakage_MAE", "Leakage_RMSE"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = leakage_compare_df.sort_values(metric, ascending=True)
    ax = sns.barplot(data=plot_df, x="Model", y=metric)
    add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{metric} Comparison After Label Shuffling")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(10, 5.5))
plot_df = leakage_compare_df.sort_values("Leakage_R2", ascending=False)
ax = sns.barplot(data=plot_df, x="Model", y="Leakage_R2")
add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.title("Leakage_R2 Comparison After Label Shuffling")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Leakage_R2_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
leakage_melt = leakage_compare_df.melt(
    id_vars="Model",
    value_vars=["Leakage_MAE", "Leakage_RMSE", "Leakage_R2"],
    var_name="Metric",
    value_name="Value"
)
ax = sns.barplot(data=leakage_melt, x="Model", y="Value", hue="Metric")
add_bar_labels(ax, fmt="{:.3f}", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.title("Leakage Test Comparison Across Models")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Severity_Leakage_Test_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# Taylor diagram
std_ref = float(np.std(y.astype(float), ddof=1))
make_taylor_diagram(std_ref, taylor_stats, os.path.join(output_path, "Severity_Taylor_Diagram.png"))

print("\nAll severity model comparison results and graphs have been saved.")