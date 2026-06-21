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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
output_path = r"E:\Farzad\AI for Damage Detection of Beams\ML\Hybrid\Results\2. Location_Model_Comparison5"
os.makedirs(output_path, exist_ok=True)

# ==============================
# INPUT FILES
# ==============================

file_rms  = os.path.join(data_path, "Damage Location (ΔRMS)_pm2percent_fixed.xlsx")
file_peak = os.path.join(data_path, "Damage Location (ΔPV)_pm2percent_fixed.xlsx")
file_cf   = os.path.join(data_path, "Damage Location (ΔCF)_pm2percent_fixed.xlsx")
file_df   = os.path.join(data_path, "Damage Location (Δf).xlsx")
file_mode = os.path.join(data_path, "Damage Location (Deltamodes)TR.xlsx")

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

y = pd.to_numeric(rms.iloc[:, 0], errors="coerce").fillna(0).astype(int)

feature_blocks = [
    rms.iloc[:, 1:].add_prefix("RMS_"),
    pv.iloc[:, 1:].add_prefix("PV_"),
    cf.iloc[:, 1:].add_prefix("CF_"),
    df_f.iloc[:, 1:].add_prefix("DF_"),
    mode.iloc[:, 1:].add_prefix("MODE_")
]

X = pd.concat(feature_blocks, axis=1)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X.columns = [f"F{i+1}_{c}" for i, c in enumerate(X.columns)]
feature_names = X.columns.tolist()

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Number of location classes:", y.nunique())
print("Class counts:\n", y.value_counts().sort_index())

# ==============================
# PCA + AUGMENTATION SETTINGS
# ==============================

PCA_COMPONENTS = min(15, X.shape[1])

USE_AUGMENTATION = True
AUG_COPIES = 1                 # 1 extra copy per training sample
NOISE_LEVEL = 0.01             # 1% of feature std
SCALE_LOW = 0.99
SCALE_HIGH = 1.01
AUG_RANDOM_STATE = 42

# ==============================
# MODELS
# ==============================

models = {
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ]),

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", LogisticRegression(
            max_iter=3000,
            C=0.3,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=16,
            learning_rate="adaptive",
            learning_rate_init=0.0005,
            max_iter=2500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42
        ))
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "ExtraTrees": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            subsample=0.8,
            max_depth=3,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "GaussianProcess": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("model", GaussianProcessClassifier(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
            random_state=42,
            max_iter_predict=100
        ))
    ])
}

# ==============================
# HELPERS
# ==============================

def augment_training_data(X_train, y_train, n_copies=1, noise_level=0.01, scale_low=0.99, scale_high=1.01, random_state=42):
    rng = np.random.RandomState(random_state)

    X_train_df = X_train.copy()
    y_train_sr = y_train.copy()

    feature_std = X_train_df.std(axis=0).replace(0, 1.0)
    augmented_X = [X_train_df]
    augmented_y = [y_train_sr]

    for _ in range(n_copies):
        scale_factors = rng.uniform(scale_low, scale_high, size=X_train_df.shape)
        noise = rng.normal(loc=0.0, scale=noise_level, size=X_train_df.shape) * feature_std.values.reshape(1, -1)

        X_aug = X_train_df.values * scale_factors + noise
        X_aug_df = pd.DataFrame(X_aug, columns=X_train_df.columns, index=X_train_df.index)

        augmented_X.append(X_aug_df)
        augmented_y.append(y_train_sr.copy())

    X_out = pd.concat(augmented_X, axis=0, ignore_index=True)
    y_out = pd.concat(augmented_y, axis=0, ignore_index=True)

    perm = rng.permutation(len(X_out))
    X_out = X_out.iloc[perm].reset_index(drop=True)
    y_out = y_out.iloc[perm].reset_index(drop=True)

    return X_out, y_out


def get_prediction_scores(estimator, X_test):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
        return np.max(proba, axis=1)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            return scores.astype(float)
        return np.max(scores, axis=1)
    pred = estimator.predict(X_test)
    return pred.astype(float)


def extract_feature_importance(estimator, full_feature_names, X_ref, y_ref, scoring_metric):
    model_step = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator

    if hasattr(estimator, "named_steps") and "pca" in estimator.named_steps:
        pca_step = estimator.named_steps["pca"]
        transformed_feature_names = [f"PC{i+1}" for i in range(pca_step.n_components_)]
    else:
        transformed_feature_names = full_feature_names

    if hasattr(model_step, "feature_importances_"):
        imp = np.asarray(model_step.feature_importances_, dtype=float)
        return pd.DataFrame({"Feature": transformed_feature_names, "Importance": imp}), "Built-in"

    if hasattr(model_step, "coef_"):
        coef = np.asarray(model_step.coef_)
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            imp = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame({"Feature": transformed_feature_names, "Importance": np.asarray(imp, dtype=float)}), "Absolute Coefficients"

    if hasattr(model_step, "coefs_") and len(model_step.coefs_) > 0:
        first_layer = np.asarray(model_step.coefs_[0])
        imp = np.mean(np.abs(first_layer), axis=1)
        return pd.DataFrame({"Feature": transformed_feature_names, "Importance": np.asarray(imp, dtype=float)}), "Mean Absolute First-Layer Weights"

    perm = permutation_importance(
        estimator, X_ref, y_ref,
        n_repeats=10, random_state=42,
        scoring=scoring_metric, n_jobs=1
    )
    return pd.DataFrame({"Feature": full_feature_names, "Importance": np.asarray(perm.importances_mean, dtype=float)}), "Permutation"


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
        pred_all = np.zeros(len(y_perm), dtype=int)

        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_data, y_perm), start=1):
            est = clone(estimator)
            X_train, X_test = X_data.iloc[train_idx].copy(), X_data.iloc[test_idx].copy()
            y_train, y_test = y_perm.iloc[train_idx].copy(), y_perm.iloc[test_idx].copy()

            if USE_AUGMENTATION:
                X_train, y_train = augment_training_data(
                    X_train, y_train,
                    n_copies=AUG_COPIES,
                    noise_level=NOISE_LEVEL,
                    scale_low=SCALE_LOW,
                    scale_high=SCALE_HIGH,
                    random_state=AUG_RANDOM_STATE + run * 100 + fold_id
                )

            est.fit(X_train, y_train)
            pred_all[test_idx] = est.predict(X_test).astype(int)

        rows.append([
            run + 1,
            accuracy_score(y_perm, pred_all),
            f1_score(y_perm, pred_all, average="weighted", zero_division=0)
        ])

    return pd.DataFrame(rows, columns=["Run", "Accuracy", "F1_weighted"])


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
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:.2f}" for c in corr_ticks])

    ax.plot(0, std_ref, "k*", markersize=14, label="Reference")

    for name, vals in model_stats.items():
        theta = np.arccos(np.clip(vals["corr"], -1, 1))
        radius = vals["std"]
        ax.scatter(theta, radius, s=80, label=name)

    ax.set_title("Taylor Diagram", pad=25, fontsize=18)
    fig.text(0.79, 0.08, "Radial axis: Standard Deviation (mm/classes)", ha="center", fontsize=10)
    fig.text(0.23, 0.92, "Angular axis: Correlation Coefficient (unitless)", ha="center", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def add_bar_labels(ax, fmt="{:.3f}", fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2., height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 4),
            textcoords="offset points"
        )

# ==============================
# CROSS VALIDATION + ALL OUTPUTS
# ==============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
fold_rows = []
all_cm = {}
all_pred_true = {}
taylor_stats = {}
learning_curve_summary = []

unique_classes = np.sort(y.unique())

for name, base_model in models.items():
    print(f"\nTraining {name} ...")

    model_folder = os.path.join(output_path, name)
    os.makedirs(model_folder, exist_ok=True)

    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        est = clone(base_model)
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        if USE_AUGMENTATION:
            X_train, y_train = augment_training_data(
                X_train, y_train,
                n_copies=AUG_COPIES,
                noise_level=NOISE_LEVEL,
                scale_low=SCALE_LOW,
                scale_high=SCALE_HIGH,
                random_state=AUG_RANDOM_STATE + fold
            )

        est.fit(X_train, y_train)
        pred = est.predict(X_test).astype(int)
        pred_score = get_prediction_scores(est, X_test)

        oof_pred[test_idx] = pred
        oof_score[test_idx] = pred_score

        abs_err = np.abs(y_test.values - pred.astype(float))
        fold_acc = accuracy_score(y_test, pred)
        fold_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        fold_mae = np.mean(abs_err)
        fold_rmse = np.sqrt(np.mean((y_test.values - pred.astype(float)) ** 2))
        fold_w100 = np.mean(abs_err <= 100) * 100
        fold_w200 = np.mean(abs_err <= 200) * 100

        fold_metrics.append([fold, fold_acc, fold_f1, fold_mae, fold_rmse, fold_w100, fold_w200])
        fold_rows.append([name, fold, fold_acc, fold_f1, fold_mae, fold_rmse, fold_w100, fold_w200])

    fold_df = pd.DataFrame(fold_metrics, columns=[
        "Fold", "Accuracy", "F1_weighted", "MAE_mm", "RMSE_mm", "Within_100mm_%", "Within_200mm_%"
    ])
    fold_df.loc[len(fold_df)] = [
        "Mean",
        fold_df["Accuracy"].mean(),
        fold_df["F1_weighted"].mean(),
        fold_df["MAE_mm"].mean(),
        fold_df["RMSE_mm"].mean(),
        fold_df["Within_100mm_%"].mean(),
        fold_df["Within_200mm_%"].mean()
    ]
    fold_df.loc[len(fold_df)] = [
        "Std",
        fold_df.iloc[:5]["Accuracy"].std(ddof=1),
        fold_df.iloc[:5]["F1_weighted"].std(ddof=1),
        fold_df.iloc[:5]["MAE_mm"].std(ddof=1),
        fold_df.iloc[:5]["RMSE_mm"].std(ddof=1),
        fold_df.iloc[:5]["Within_100mm_%"].std(ddof=1),
        fold_df.iloc[:5]["Within_200mm_%"].std(ddof=1)
    ]
    fold_df.to_excel(os.path.join(model_folder, f"{name}_KFold_Results.xlsx"), index=False)

    oof_acc = accuracy_score(y, oof_pred)
    oof_f1 = f1_score(y, oof_pred, average="weighted", zero_division=0)
    abs_err = np.abs(y.values - oof_pred.astype(float))
    oof_mae = float(np.mean(abs_err))
    oof_rmse = float(np.sqrt(np.mean((y.values - oof_pred.astype(float)) ** 2)))
    oof_w100 = float(np.mean(abs_err <= 100) * 100)
    oof_w200 = float(np.mean(abs_err <= 200) * 100)

    cm = confusion_matrix(y, oof_pred, labels=unique_classes)
    all_cm[name] = cm
    all_pred_true[name] = (y.values.copy(), oof_pred.astype(float).copy())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"Location Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Confusion_Matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # learning curve remains on original data for clean comparison
    train_sizes, train_scores, val_scores = learning_curve(
        clone(base_model), X, y,
        cv=cv,
        scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    lc_df = pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_F1_weighted_Mean": train_mean,
        "Train_F1_weighted_Std": train_std,
        "Validation_F1_weighted_Mean": val_mean,
        "Validation_F1_weighted_Std": val_std
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
    plt.ylabel("F1-weighted")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Learning_Curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fitted_full = clone(base_model)
    fit_X, fit_y = X.copy(), y.copy()
    if USE_AUGMENTATION:
        fit_X, fit_y = augment_training_data(
            fit_X, fit_y,
            n_copies=AUG_COPIES,
            noise_level=NOISE_LEVEL,
            scale_low=SCALE_LOW,
            scale_high=SCALE_HIGH,
            random_state=AUG_RANDOM_STATE + 999
        )
    fitted_full.fit(fit_X, fit_y)

    imp_df, importance_method = extract_feature_importance(fitted_full, feature_names, X, y, "f1_weighted")
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df.to_excel(os.path.join(model_folder, f"{name}_Feature_Importance.xlsx"), index=False)
    save_top_importance_plot(
        imp_df,
        f"Top Feature Importance - {name} ({importance_method})",
        os.path.join(model_folder, f"{name}_Top_Feature_Importance.png"),
        top_n=15
    )

    perm = permutation_importance(
        fitted_full, X, y,
        n_repeats=10,
        random_state=42,
        scoring="f1_weighted",
        n_jobs=1
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

    plt.figure(figsize=(7, 7))
    plt.scatter(y.values, oof_pred.astype(float), alpha=0.7)
    min_val = min(y.min(), oof_pred.min())
    max_val = max(y.max(), oof_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--")
    plt.xlabel("True Location (mm/class)")
    plt.ylabel("Predicted Location (mm/class)")
    plt.title(f"Predicted vs True Location - {name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Predicted_vs_True.png"), dpi=300, bbox_inches="tight")
    plt.close()

    error_df = pd.DataFrame({
        "True_Location": y.values,
        "Predicted_Class": oof_pred,
        "Absolute_Error": abs_err
    })
    error_df.to_excel(os.path.join(model_folder, f"{name}_Predictions_and_Errors.xlsx"), index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(abs_err, bins=20)
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.title(f"Location Error Histogram - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Error_Histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(abs_err)), abs_err, marker="o", linestyle="-")
    plt.xlabel("Sample Index")
    plt.ylabel("Absolute Error")
    plt.title(f"Location Error by Sample - {name}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Error_By_Sample.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fold_only = fold_df.iloc[:5].copy()
    x = np.arange(1, 6)
    plt.figure(figsize=(8, 5))
    plt.plot(x, fold_only["Accuracy"], marker="o", label="Accuracy")
    plt.plot(x, fold_only["F1_weighted"], marker="o", label="F1-weighted")
    plt.plot(x, fold_only["MAE_mm"], marker="o", label="MAE")
    plt.plot(x, fold_only["RMSE_mm"], marker="o", label="RMSE")
    plt.xlabel("Fold")
    plt.ylabel("Metric Value")
    plt.title(f"Cross-Validation Fold Performance - {name}")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Fold_Performance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    leakage_df = permutation_leakage_test(base_model, X, y, cv, n_runs=10)
    leakage_df.to_excel(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.xlsx"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(leakage_df["Run"], leakage_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(leakage_df["Run"], leakage_df["F1_weighted"], marker="s", label="F1-weighted")
    plt.xlabel("Permutation Run")
    plt.ylabel("Score")
    plt.title(f"Permutation Leakage Test - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, f"{name}_Permutation_Leakage_Test.png"), dpi=300, bbox_inches="tight")
    plt.close()

    final_summary_df = pd.DataFrame({
        "Metric": [
            "OOF Accuracy", "OOF F1_weighted", "OOF MAE", "OOF RMSE",
            "OOF Within_100_%", "OOF Within_200_%",
            "CV Accuracy Mean", "CV Accuracy Std",
            "CV F1_weighted Mean", "CV F1_weighted Std",
            "CV MAE Mean", "CV MAE Std",
            "CV RMSE Mean", "CV RMSE Std",
            "Permutation Accuracy Mean", "Permutation F1_weighted Mean"
        ],
        "Value": [
            oof_acc, oof_f1, oof_mae, oof_rmse, oof_w100, oof_w200,
            fold_df.iloc[:5]["Accuracy"].mean(), fold_df.iloc[:5]["Accuracy"].std(ddof=1),
            fold_df.iloc[:5]["F1_weighted"].mean(), fold_df.iloc[:5]["F1_weighted"].std(ddof=1),
            fold_df.iloc[:5]["MAE_mm"].mean(), fold_df.iloc[:5]["MAE_mm"].std(ddof=1),
            fold_df.iloc[:5]["RMSE_mm"].mean(), fold_df.iloc[:5]["RMSE_mm"].std(ddof=1),
            leakage_df["Accuracy"].mean(), leakage_df["F1_weighted"].mean()
        ]
    })
    final_summary_df.to_excel(os.path.join(model_folder, f"{name}_Final_Summary.xlsx"), index=False)

    summary_rows.append([
        name,
        oof_acc,
        oof_f1,
        oof_mae,
        oof_rmse,
        oof_w100,
        oof_w200,
        fold_df.iloc[:5]["Accuracy"].mean(),
        fold_df.iloc[:5]["Accuracy"].std(ddof=1),
        fold_df.iloc[:5]["F1_weighted"].mean(),
        fold_df.iloc[:5]["F1_weighted"].std(ddof=1),
        leakage_df["Accuracy"].mean(),
        leakage_df["F1_weighted"].mean(),
        importance_method
    ])

    y_ref = y.astype(float).values
    y_model = oof_pred.astype(float)
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
    "Model", "Accuracy", "F1_weighted", "MAE", "RMSE", "Within_100_%", "Within_200_%",
    "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_F1_weighted_Mean", "CV_F1_weighted_Std",
    "Permutation_Accuracy_Mean", "Permutation_F1_weighted_Mean", "Feature_Importance_Method"
]).sort_values(["F1_weighted", "Accuracy"], ascending=[False, False]).reset_index(drop=True)
results_df.to_excel(os.path.join(output_path, "Location_Model_Comparison.xlsx"), index=False)

fold_compare_df = pd.DataFrame(fold_rows, columns=[
    "Model", "Fold", "Accuracy", "F1_weighted", "MAE_mm", "RMSE_mm", "Within_100mm_%", "Within_200mm_%"
])
fold_compare_df.to_excel(os.path.join(output_path, "Location_All_Models_Foldwise_Results.xlsx"), index=False)

print("\nOverall summary:")
print(results_df)

# ==============================
# COMPARISON PLOTS
# ==============================

for metric in ["Accuracy", "F1_weighted", "Within_100_%", "Within_200_%"]:
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

perm_drop_df = results_df[[
    "Model", "Accuracy", "F1_weighted",
    "Permutation_Accuracy_Mean", "Permutation_F1_weighted_Mean"
]].copy()

perm_drop_df["Accuracy_Drop"] = perm_drop_df["Accuracy"] - perm_drop_df["Permutation_Accuracy_Mean"]
perm_drop_df["F1_weighted_Drop"] = perm_drop_df["F1_weighted"] - perm_drop_df["Permutation_F1_weighted_Mean"]
perm_drop_df.to_excel(os.path.join(output_path, "Location_Permutation_Drop_Comparison.xlsx"), index=False)

for metric in ["Accuracy_Drop", "F1_weighted_Drop"]:
    plt.figure(figsize=(10, 5.5))
    plot_df = perm_drop_df.sort_values(metric, ascending=False)
    ax = sns.barplot(data=plot_df, x="Model", y=metric)
    add_bar_labels(ax, fmt="{:.3f}", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Performance Drop")
    plt.title(f"{metric.replace('_', ' ')} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

for metric in ["Accuracy", "F1_weighted", "MAE_mm", "RMSE_mm"]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fold_compare_df, x="Model", y=metric)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Fold-wise {metric} Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"Foldwise_{metric}_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

heatmap_df = results_df.set_index("Model")[["Accuracy", "F1_weighted", "MAE", "RMSE", "Within_100_%", "Within_200_%"]]
plt.figure(figsize=(9, 6))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title("Location Model Performance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_Model_Performance_Heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

radar_metrics = ["Accuracy", "F1_weighted", "Within_100_%", "Within_200_%"]
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
ax.set_title("Radar Comparison of Location Models")
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_Radar_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

n_models = len(models)
n_cols = 4
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()
for ax, (name, cm) in zip(axes, all_cm.items()):
    sns.heatmap(cm, cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
for ax in axes[len(all_cm):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_All_Models_Confusion_Matrices.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 6))
for name, base_model in models.items():
    train_sizes, _, val_scores = learning_curve(
        clone(base_model), X, y, cv=cv, scoring="f1_weighted",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=1
    )
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="o", label=name)
plt.xlabel("Training Samples")
plt.ylabel("Validation F1-weighted")
plt.title("Learning Curve Comparison - Location Models")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_Learning_Curve_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

lc_compare_df = pd.DataFrame(
    learning_curve_summary,
    columns=["Model", "Training_Final", "Validation_Final"]
)
lc_compare_df.to_excel(os.path.join(output_path, "Location_Training_Validation_Comparison.xlsx"), index=False)

plot_df = lc_compare_df.melt(
    id_vars="Model",
    value_vars=["Training_Final", "Validation_Final"],
    var_name="Type",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Type")
add_bar_labels(ax, fmt="{:.3f}", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.title("Training vs Validation F1-weighted Comparison")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_Training_vs_Validation_F1_weighted_Comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()
for ax, (name, (y_true_vals, y_pred_vals)) in zip(axes, all_pred_true.items()):
    ax.scatter(y_true_vals, y_pred_vals, alpha=0.6)
    min_val = min(np.min(y_true_vals), np.min(y_pred_vals))
    max_val = max(np.max(y_true_vals), np.max(y_pred_vals))
    ax.plot([min_val, max_val], [min_val, max_val], "--")
    ax.set_title(name)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
for ax in axes[len(all_pred_true):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "Location_All_Models_Predicted_vs_True.png"), dpi=300, bbox_inches="tight")
plt.close()

std_ref = float(np.std(y.astype(float), ddof=1))
make_taylor_diagram(std_ref, taylor_stats, os.path.join(output_path, "Location_Taylor_Diagram.png"))

print("\nAll location model comparison results and graphs have been saved.")