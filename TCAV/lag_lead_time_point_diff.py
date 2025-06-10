import __fix_relative_imports  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt

from mscEidalVesetrud.TCAV.compute_r2_on_true_model import get_lags_leads

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

_, r2_avg_std_time = get_lags_leads()

plt.figure(constrained_layout=True, figsize=(3.5, 2.5))


linestyles = [(0, (1, 1)), (0, (3, 2)), (0, (3, 1, 1, 1)), "-"]
for i, (layer_name, data) in enumerate(r2_avg_std_time.items()):
    time_steps = np.array(list(data.keys()))
    avg_r2 = np.array([v[0] for v in data.values()])
    std = np.array([v[1] for v in data.values()])

    print(f"{np.average(std):.04f}")

    plt.plot(
        time_steps,
        avg_r2,
        label=f"Layer {layer_name[-1]}",
        color=f"C{i}",
        ls=linestyles[i],
    )

    # plt.errorbar(
    #     time_steps,  # Select a subset of time steps
    #     avg_r2,
    #     yerr=std,
    #     fmt="none",  # Don't plot markers at the error bars
    #     capsize=3,
    #     color=f"C{i}",
    #     alpha=0.7,
    # )

plt.xlabel("Time relative to center (hours)")
plt.ylabel("Average $R^2$")
# plt.title("")
plt.legend(ncol=2, loc="lower center")
plt.grid(True)
plt.ylim(0, 1)
plt.savefig("lag_lead_time_diff.pdf", format="pdf")
plt.show()
