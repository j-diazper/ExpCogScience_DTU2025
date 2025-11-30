# -----------------------
# IMPORTS
# -----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, rankdata
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# -----------------------
# LOAD DATA
# -----------------------
# Adjust path to your data folder
data_folder = "data"
all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and "keylog" not in f]

# Load all participants
dfs = []
for f in all_files:
    df = pd.read_csv(os.path.join(data_folder, f))
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Compute accuracy per participant
accuracy_df = data.groupby(['ParticipantID', 'Condition']).agg(
    Accuracy=('Accuracy', 'mean'),
    Mood=('Mood', 'first'),
    Fatigue=('Fatigue', 'first')
).reset_index()

# -----------------------
# DESCRIPTIVE STATS
# -----------------------
desc_stats = accuracy_df.describe()
desc_stats_tex = desc_stats.to_latex(float_format="%.3f")
with open("descriptive_stats.tex", "w") as f:
    f.write(desc_stats_tex)

# -----------------------
# BOXPLOTS
# -----------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='Condition', y='Accuracy', data=accuracy_df)
plt.title('Accuracy by Condition')
plt.savefig("accuracy_boxplot.png", dpi=300)
plt.close()

# -----------------------
# ANCOVA
# -----------------------
formula = 'Accuracy ~ C(Condition) + Mood + Fatigue'
model = ols(formula, data=accuracy_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table_tex = anova_table.to_latex(float_format="%.3f")
with open("ancova_table.tex", "w") as f:
    f.write(anova_table_tex)

# Check assumptions
# Shapiro-Wilk on residuals
shapiro_test = shapiro(model.resid)
# Levene test for equal variances
groups = [group["Accuracy"].values for name, group in accuracy_df.groupby("Condition")]
levene_test = levene(*groups)

# Scatter plots for covariates vs accuracy
plt.figure(figsize=(6,4))
sns.scatterplot(x='Mood', y='Accuracy', hue='Condition', data=accuracy_df)
sns.scatterplot(x='Fatigue', y='Accuracy', hue='Condition', data=accuracy_df)
plt.savefig("covariate_scatterplots.png", dpi=300)
plt.close()

# -----------------------
# RANKED ANCOVA
# -----------------------
ranked_df = accuracy_df.copy()
ranked_df['Accuracy_rank'] = rankdata(ranked_df['Accuracy'])
ranked_model = ols('Accuracy_rank ~ C(Condition) + Mood + Fatigue', data=ranked_df).fit()
ranked_anova = sm.stats.anova_lm(ranked_model, typ=2)
ranked_anova_tex = ranked_anova.to_latex(float_format="%.3f")
with open("ranked_ancova_table.tex", "w") as f:
    f.write(ranked_anova_tex)

print("Analysis complete. Figures and LaTeX tables generated:")
print("- accuracy_boxplot.png")
print("- covariate_scatterplots.png")
print("- descriptive_stats.tex")
print("- ancova_table.tex")
print("- ranked_ancova_table.tex")
