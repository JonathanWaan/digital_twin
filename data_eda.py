import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
# Load dataset
data = pd.read_csv("data/sentences.csv")

# Filter for Hume
hume_data = data[data["author"] == "Hume"].copy()

# Assume you already have Sentence-BERT embeddings
# Replace with your actual embedding loading method
# Example: np.ndarray of shape (num_hume_sentences, 768)
import numpy as np
import pickle

# Example: load precomputed embeddings
# with open("hume_embeddings.pkl", "rb") as f:
#     hume_embs = pickle.load(f)
# Placeholder if needed
# hume_embs = np.random.randn(len(hume_data), 768)
# Use real embeddings in your workflow
embeddings = torch.load("data/gemini_768_sentence.pt")
hume_embs = embeddings[data["author"] == "Hume"].numpy()
assert len(hume_data) == len(hume_embs)

# # t-SNE projection
# print("running t-sne projection")
# tsne = TSNE(n_components=2, random_state=42, perplexity=30)
# hume_2d = tsne.fit_transform(hume_embs)
# print("finished t-sne projection")
# # Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(hume_2d[:, 0], hume_2d[:, 1], s=10, alpha=0.6, c="steelblue")
# plt.title("t-SNE of Hume Sentence Embeddings")
# plt.xlabel("t-SNE-1")
# plt.ylabel("t-SNE-2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # save plot
# plt.savefig("results/plot/hume_tsne_plot.png", dpi=300)


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Add sentence length column
data["char_len"] = data["sentence"].apply(len)

# Subplot setup
authors = data["author"].unique()
n = len(authors)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharex=True, sharey=True)

color = sns.color_palette("colorblind")[4]

# Plot each author in a shared-row layout
for ax, author in zip(axes, authors):
    subset = data[data["author"] == author]
    sns.histplot(
        data=subset,
        x="char_len",
        bins=50,
        stat="density",
        alpha=0.6,
        color=color,
        ax=ax
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(f"{author}")
    ax.set_xlabel("Char Len")
    ax.set_ylabel("Percentage")
    ax.grid(True)

plt.tight_layout()
plt.savefig("results/plot/sentence_length_density_row_shared_y.png", dpi=300)
plt.show()
