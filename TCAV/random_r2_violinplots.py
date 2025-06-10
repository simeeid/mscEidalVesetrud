import __fix_relative_imports  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import shapiro

from mscEidalVesetrud.TCAV.compute_r2_on_true_model import load_r2_sign_test

# Set the font globally
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Computer Modern Roman"
# Use the below instead if you do not have LaTeX installed
# plt.rcParams["font.family"] = "cmr10"
# plt.rcParams["axes.formatter.use_mathtext"] = True

plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"


_, _, random_r2_scores = load_r2_sign_test()
# Assuming your four lists are list1, list2, list3, list4
data = list(random_r2_scores.values())
time_points = list(random_r2_scores.keys())

# plt.figure(figsize=(5.5, 4), constrained_layout=True)
# plt.boxplot(data, labels=time_points)
# plt.xlabel("Network Layer")
# plt.ylabel("$R^2$")
# plt.title("Distribution Evolution Over Layers")
# # plt.grid(axis="y", linestyle="--")
# plt.ylim(0, 1)
# plt.show()

for layer, rnd_r2 in random_r2_scores.items():
    print(layer, np.average(rnd_r2))

data = {
    "Layer": np.ravel(
        [[f"{name[-1]}"] * len(data) for name, data in random_r2_scores.items()]
    ),
    "R2": np.concat(data),
}
df = pd.DataFrame(data)

plt.figure(figsize=(3.6, 2.4), constrained_layout=True)
sns.violinplot(x="Layer", y="R2", data=df, inner="box")
# sns.stripplot(x="Layer", y="R2", data=df, color="black", alpha=0.5, dodge=True, size=2)
plt.xlabel("Network Layer")
plt.ylabel("$R^2$", rotation=0, labelpad=8)
plt.title("Random Concept Distributions")
plt.ylim(-0.05, 1)
plt.grid(axis="y", linestyle="--")
plt.savefig("r2_random_violin_plot.pdf", format="pdf")
plt.show()


for name, data in random_r2_scores.items():
    statistic, pvalue = shapiro(data)
    print(f"{name}, {statistic:.04f}, {pvalue:.04e}")
