import matplotlib.pyplot as plt

template_set = False
population_set = True

if template_set:
    full_title = "Template_Set"

    x = [1, 2, 3, 4, 5, 6, 7]

    chains = [
        [0.25287, 0.21389, 0.23088, 0.22167, 0.25141, 0.24628, 0.24842],
        [0.35424, 0.34687, 0.34952, 0.35058, 0.35368, 0.35324, 0.35230],
        [0.44245, 0.43785, 0.43955, 0.44167, 0.44425, 0.44059, 0.44418],
        [0.70531, 0.68032, 0.69860, 0.69847, 0.70712, 0.70423, 0.70229],
        [0.92173, 0.89721, 0.91642, 0.93412, 0.92267, 0.92614, 0.91983],
    ]

    sigma = [
        [0.00408, 0.00236, 0.00303, 0.00229, 0.00400, 0.00328, 0.00366],
        [0.00117, 0.00107, 0.00117, 0.00115, 0.00113, 0.00144, 0.00118],
        [0.00164, 0.00153, 0.00165, 0.00157, 0.00158, 0.00176, 0.00167],
        [0.00185, 0.00186, 0.00184, 0.00192, 0.00195, 0.00186, 0.00191],
        [0.00230, 0.00228, 0.00236, 0.00241, 0.00238, 0.00236, 0.00236],
    ]

    labels = ["PCA", "50_1", "50_2", "50_3", "100_1", "100_2", "100_3"]

if population_set:
    full_title = "Population_Set"

    x = [1, 2, 3, 4]

    chains = [
        [0.25287, 0.26111, 0.26044, 0.25571],
        [0.35424, 0.35248, 0.35529, 0.35227],
        [0.44245, 0.44328, 0.44438, 0.44118],
        [0.70531, 0.70907, 0.70814, 0.708529],
        [0.92173, 0.92337, 0.92449, 0.92449],
    ]

    sigma = [
        [0.00408, 0.00435, 0.00416, 0.00420],
        [0.00117, 0.00122, 0.00127, 0.00122],
        [0.00164, 0.00171, 0.00168, 0.00168],
        [0.00185, 0.00189, 0.00189, 0.00194],
        [0.00230, 0.00237, 0.00230, 0.00230],
    ]

    labels = ["PCA", "set2", "set3", "set4"]

titles = [
    "$0.1 < z_{BPZ} < 0.3$",
    "$0.3 < z_{BPZ} < 0.5$",
    "$0.5 < z_{BPZ} < 0.7$",
    "$0.7 < z_{BPZ} < 0.9$",
    "$0.9 < z_{BPZ} < 1.2$",
]

fig, ax = plt.subplots(2, 3, sharex=False, sharey=False)
for i in range(6):
    row = i // 3
    col = i % 3
    ax[row, col].set_xticks(x)
    ax[row, col].yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    if i < 5:
        ax[row, col].errorbar(x[1:], chains[i][1:], yerr=sigma[i][1:], fmt="o", c="C0")
        ax[row, col].errorbar(x[0], chains[i][0], yerr=sigma[i][0], fmt="o", c="C1")
        ax[row, col].axhspan(
            chains[i][0] - sigma[i][0], chains[i][0] + sigma[i][0], facecolor="0.9"
        )
        ax[row, col].set_title(r"{}".format(titles[i]))
    if i == 3:
        ax[row, col].set_xticklabels(labels, rotation=50)
    else:
        ax[row, col].set_xticklabels([])
    if i != 3:
        ax[row, col].tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

fig.subplots_adjust(hspace=0.15)
fig.subplots_adjust(wspace=0.35)
fig.delaxes(ax[1, 2])
fig.text(0.5, 0.02, full_title, ha="center", va="center")
fig.text(0.04, 0.5, r"$\bar{z}$", ha="center", va="center")

plt.show()
