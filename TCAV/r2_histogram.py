import __fix_relative_imports  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from mscEidalVesetrud.TCAV.compute_r2_on_true_model import get_r2_per_layer

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


r2_per_layer = get_r2_per_layer()


plt.figure(constrained_layout=True, figsize=(5.5, 3.5))

bins = np.arange(-0.1, 1.05, 0.1)  # Define bins from 0 to 1


linestyles = [(0, (1, 2)), (0, (3, 3)), (0, (3, 2, 1, 2)), "-"]
for i, (name, data) in enumerate(r2_per_layer.items()):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(
        bin_centers,
        hist,
        label=f"Layer {name[-1]}",
        color=f"C{i}",
        ls=linestyles[i],
    )

plt.xlabel("$R^2$")
plt.ylabel("Number of Baseline Features")
plt.title("Baseline Features $R^2$ Distribution")
plt.xticks(np.arange(-0.1, 1.1, 0.1))
plt.legend()
plt.grid(True)
plt.xlim(-0.1, 1.0)
plt.yticks(range(0, 51, 5))
plt.savefig("r2_step_plot.pdf", format="pdf")
plt.show()
