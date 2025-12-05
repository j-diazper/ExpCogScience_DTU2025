# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
import os

sns.set(style="whitegrid")

# =====================================================
# LOAD TRIAL-LEVEL DATA
# =====================================================
data_folder = "DataNew"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and "keylog" not in f]

dfs = []
for f in csv_files:
    df = pd.read_csv(os.path.join(data_folder, f))
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# =====================================================
# LOAD QUESTIONNAIRE
# =====================================================
questionnaire_path = "questionnaire.csv"
questionnaire = pd.read_csv(questionnaire_path)

# If questionnaire is incorrectly formatted (1 column), attempt to split
if questionnaire.shape[1] == 1:
    print("⚠ Detected single-column questionnaire file. Attempting to split automatically...")
    questionnaire = questionnaire.iloc[:, 0].str.split(",", expand=True)
    questionnaire.columns = ["ParticipantID", "Mood", "Fatigue", "Smell"]

questionnaire["ParticipantID"] = questionnaire["ParticipantID"].astype(str)
questionnaire[["Mood", "Fatigue", "Smell"]] = questionnaire[["Mood", "Fatigue", "Smell"]].apply(pd.to_numeric)

print(questionnaire)
# =====================================================
# COMPUTE ACCURACY PER PARTICIPANT
# =====================================================
accuracy_df = data.groupby(["ParticipantID", "Condition"], as_index=False).agg(
    Accuracy=("Accuracy", "mean")
)
accuracy_df["ParticipantID"] = accuracy_df["ParticipantID"].astype(str)
print(accuracy_df)
# Merge mood/fatigue/smell
accuracy_df = accuracy_df.merge(questionnaire, on="ParticipantID", how="left")


# =====================================================
# DESCRIPTIVE STATISTICS BY CONDITION (FOR PAPER TABLE)
# =====================================================
vars_for_desc = ["Accuracy", "Mood", "Fatigue", "Smell"]

desc_by_cond = (
    accuracy_df
    .groupby("Condition")[vars_for_desc]
    .agg(["mean", "std", "count"])
)

# Flatten MultiIndex columns for nicer LaTeX output
desc_by_cond.columns = [
    f"{var}_{stat}" for var, stat in desc_by_cond.columns
]

desc_by_cond.to_latex(
    "descriptive_by_condition.tex",
    float_format="%.3f",
    caption="Descriptive statistics by condition (Mean, SD, and N).",
    label="tab:descriptives_condition"
)

print("\n===== DESCRIPTIVES BY CONDITION =====")
print(desc_by_cond)

# =====================================================
# CORRELATION MATRIX (FOR APPENDIX OR RESULTS)
# =====================================================
corr_vars = ["Accuracy", "Mood", "Fatigue", "Smell"]
corr_matrix = accuracy_df[corr_vars].corr()

corr_matrix.to_latex(
    "correlation_matrix.tex",
    float_format="%.3f",
    caption="Correlation matrix for accuracy and covariates.",
    label="tab:correlations"
)

print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)


# =====================================================
# SAVE DESCRIPTIVE STATISTICS
# =====================================================
desc_stats = accuracy_df.describe()
desc_stats.to_latex("descriptive_stats.tex", float_format="%.3f")

print("\n===== DESCRIPTIVE STATS =====")
print(desc_stats)


# =====================================================
# BOXPLOT
# =====================================================
plt.figure(figsize=(6,4))
sns.boxplot(x="Condition", y="Accuracy", data=accuracy_df)
sns.swarmplot(x="Condition", y="Accuracy", data=accuracy_df, color="black", size=4)
plt.title("Accuracy by Condition")
plt.ylim(0,1)
plt.savefig("plot_accuracy_boxplot.png", dpi=300)
plt.close()


# =====================================================
# DIAGNOSTIC PLOTS (for normality/linearity)
# =====================================================
base_model = ols("Accuracy ~ C(Condition) + Mood + Fatigue + Smell", data=accuracy_df).fit()

# Residual histogram
plt.figure(figsize=(5,4))
plt.hist(base_model.resid, bins=15)
plt.title("Residual Histogram")
plt.savefig("diagnostic_residual_hist.png", dpi=300)
plt.close()

# Q-Q plot
plt.figure(figsize=(5,4))
qqplot(base_model.resid, line='s')
plt.title("Q–Q Plot of Residuals")
plt.savefig("diagnostic_qqplot.png", dpi=300)
plt.close()

# Residuals vs fitted
plt.figure(figsize=(5,4))
plt.scatter(base_model.fittedvalues, base_model.resid)
plt.axhline(0, color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted")
plt.savefig("diagnostic_resid_vs_fitted.png", dpi=300)
plt.close()

# Linearity plots
for cov in ["Mood", "Fatigue", "Smell"]:
    plt.figure(figsize=(5,4))
    sns.regplot(x=cov, y="Accuracy", data=accuracy_df, ci=None, scatter_kws={"s":40})
    plt.title(f"Linearity Check: Accuracy vs. {cov}")
    plt.savefig(f"diagnostic_linearity_{cov}.png", dpi=300)
    plt.close()


# =====================================================
# ASSUMPTION TESTS
# =====================================================
print("\n===== ASSUMPTIONS =====")

# Normality (Shapiro–Wilk)
shapiro_stat, shapiro_p = shapiro(base_model.resid)
print(f"Shapiro p = {shapiro_p:.4f}")

# Homogeneity of variances (Levene)
groups = [group["Accuracy"].values for name, group in accuracy_df.groupby("Condition")]
levene_stat, levene_p = levene(*groups)
print(f"Levene p = {levene_p:.4f}")

# Homogeneity of regression slopes
interaction_model = ols(
    "Accuracy ~ C(Condition)*Mood + C(Condition)*Fatigue + C(Condition)*Smell",
    data=accuracy_df
).fit()

interaction_anova = sm.stats.anova_lm(interaction_model, typ=2)
interaction_pvalues = interaction_anova.loc[[i for i in interaction_anova.index if ":" in i], "PR(>F)"]

print("\nInteraction p-values:")
print(interaction_pvalues)

# Save assumption results
with open("assumption_tests.txt", "w") as f:
    f.write("ASSUMPTION TEST RESULTS\n")
    f.write(f"Shapiro p = {shapiro_p}\n")
    f.write(f"Levene p = {levene_p}\n")
    f.write("Interaction p-values:\n")
    f.write(interaction_pvalues.to_string())


# =====================================================
# DECISION: PARAMETRIC OR RANKED ANCOVA
# =====================================================
use_ranked = False

if shapiro_p <= 0.05:
    use_ranked = True
if levene_p <= 0.05:
    use_ranked = True
if np.any(interaction_pvalues <= 0.05):
    use_ranked = True


# =====================================================
# RUN FINAL MODEL
# =====================================================
if not use_ranked:
    print("\nAssumptions satisfied — Running PARAMETRIC ANCOVA")
    final_model = base_model
    final_anova = sm.stats.anova_lm(final_model, typ=2)

else:
    print("\nAssumptions violated — Running RANKED ANCOVA")
    ranked_df = accuracy_df.copy()
    ranked_df["Accuracy_rank"] = rankdata(ranked_df["Accuracy"])

    final_model = ols(
        "Accuracy_rank ~ C(Condition) + Mood + Fatigue + Smell",
        data=ranked_df
    ).fit()
    final_anova = sm.stats.anova_lm(final_model, typ=2)


# =====================================================
# EFFECT SIZES (PARTIAL ETA SQUARED)
# =====================================================
final_anova["eta_sq_partial"] = final_anova["sum_sq"] / (final_anova["sum_sq"] + final_anova.loc["Residual", "sum_sq"])


# =====================================================
# SAVE ANCOVA RESULTS (CLEANED FOR LaTeX)
# =====================================================
anova_display = final_anova.copy()

# Rename index for nicer display
anova_display.index = [
    idx.replace("C(Condition)", "Condition")
        .replace("Mood", "Mood")
        .replace("Fatigue", "Fatigue")
        .replace("Smell", "Smell")
        .replace("Residual", "Error")
    for idx in anova_display.index
]

# Keep relevant columns in nice order
cols_to_keep = ["df", "sum_sq", "F", "PR(>F)", "eta_sq_partial"]
anova_display = anova_display[cols_to_keep]
anova_display.columns = ["df", "SS", "F", "p", r"$\eta^2_p$"]

anova_display.to_latex(
    "final_ancova_table.tex",
    float_format="%.3f",
    caption="ANCOVA results for accuracy with Condition, Mood, Fatigue, and Smell as predictors.",
    label="tab:ancova"
)

print("\n===== FINAL ANCOVA TABLE (CLEANED) =====")
print(anova_display)
