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
data_folder = "data"
all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and "keylog" not in f]

# Load all participants
dfs = []
for f in all_files:
    df = pd.read_csv(os.path.join(data_folder, f))
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# -----------------------
# LOAD QUESTIONNAIRE DATA
# -----------------------
questionnaire_path = "questionnaire.csv"
questionnaire = pd.read_csv(questionnaire_path)
# Split the single column into four separate columns

# -----------------------
# ACCURACY PER PARTICIPANT
# -----------------------
accuracy_df = data.groupby(['ParticipantID', 'Condition'], as_index=False).agg(
    Accuracy=('Accuracy', 'mean')
)

# Merge questionnaire ratings (Mood, Fatigue, Smell)
accuracy_df = accuracy_df.merge(questionnaire, on="ParticipantID", how="left")


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

import numpy as np

# -----------------------
# CHECK ASSUMPTIONS
# -----------------------

# Base ANCOVA model (no interactions)
base_model = ols('Accuracy ~ C(Condition) + Mood + Fatigue + Smell', data=accuracy_df).fit()

# 1. Normality of residuals
shapiro_stat, shapiro_p = shapiro(base_model.resid)

# 2. Homogeneity of variances (Levene)
groups = [group["Accuracy"].values for name, group in accuracy_df.groupby("Condition")]
levene_stat, levene_p = levene(*groups)

# 3. Homogeneity of regression slopes
interaction_model = ols(
    'Accuracy ~ C(Condition)*Mood + C(Condition)*Fatigue + C(Condition)*Smell',
    data=accuracy_df
).fit()

interaction_anova = sm.stats.anova_lm(interaction_model, typ=2)

# Extract p-values for interaction terms
interaction_pvalues = interaction_anova.loc[
    [row for row in interaction_anova.index if "*" in row],
    "PR(>F)"
]

# -----------------------
# DECISION LOGIC
# -----------------------

use_ranked = False

if shapiro_p <= 0.05:
    print("Residuals violate normality → ranked ANCOVA will be used.")
    use_ranked = True

if levene_p <= 0.05:
    print("Variance is not homogeneous → ranked ANCOVA will be used.")
    use_ranked = True

if np.any(interaction_pvalues <= 0.05):
    print("Covariate × Condition interaction detected → slopes not homogeneous → ranked ANCOVA will be used.")
    use_ranked = True

# -----------------------
# RUN SELECTED MODEL
# -----------------------

if not use_ranked:
    print("Assumptions OK → running standard ANCOVA.")
    final_model = base_model
    final_anova = sm.stats.anova_lm(base_model, typ=2)

else:
    print("Running ranked ANCOVA.")
    ranked_df = accuracy_df.copy()
    ranked_df['Accuracy_rank'] = rankdata(ranked_df['Accuracy'])
    final_model = ols('Accuracy_rank ~ C(Condition) + Mood + Fatigue + Smell', data=ranked_df).fit()
    final_anova = sm.stats.anova_lm(final_model, typ=2)

# Save the final ANOVA table
final_anova.to_latex("final_ancova_table.tex", float_format="%.3f")
