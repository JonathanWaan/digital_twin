import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rq1 = pd.read_csv(
    "results/experiment_results/rq1_results_diff_k_author_cos_sim.csv"
)

# Use the ColorBrewer colorblind palette
palette = sns.color_palette("colorblind")

plt.figure(figsize=(7, 5))

for i, method in enumerate(rq1['method'].unique()):
    df_method = rq1[rq1['method'] == method]

    df_grouped = (
        df_method
        .groupby('k', as_index=False)
        .mean(numeric_only=True)
        .sort_values('k')
    )

    plt.plot(
        df_grouped['k'],
        df_grouped['mean_query_cos_sim'],
        marker='o',
        linewidth=2,
        color=palette[i % len(palette)],
        label=method
    )

plt.xlabel('k')
plt.ylabel('Mean Query Cosine Similarity')
plt.legend(title='Method')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.savefig(
    "results/plot/rq1_mean_query_cos_sim_vs_k.png",
    dpi=300,
)

breakpoint()
