import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

LLM_label_package = torch.load('results/experiment_results/LLM_label_package_1_numpy_topk_emb.pt',weights_only=False)
LLM_label = torch.load('results/experiment_results/llm_scored_sentences.pt',weights_only=False)
LLM_label = torch.load('results/experiment_results/llm_scored_sentences_topical_relavence.pt',weights_only=False)

scores = LLM_label_package['doc_emb'] @  LLM_label_package['query_emb']

scores_LLM = [res['score'] for res in LLM_label]
# convert to numpy
cos_sim = np.asarray(scores)
llm_scores = np.asarray(scores_LLM)

assert len(cos_sim) == len(llm_scores), "Length mismatch!"

plt.figure(figsize=(6, 5))
plt.scatter(cos_sim, llm_scores, alpha=0.5)

# linear trend line
coef = np.polyfit(cos_sim, llm_scores, deg=1)
x_fit = np.linspace(cos_sim.min(), cos_sim.max(), 100)
y_fit = coef[0] * x_fit + coef[1]
plt.plot(x_fit, y_fit)

plt.xlabel("Cosine Similarity (query · doc)")
plt.ylabel("LLM Relevance Score")
plt.title("LLM Score vs Cosine Similarity")
plt.tight_layout()
plt.show()
# save plot
plt.savefig("results/plot/LLM_score_vs_cosine_similarity.png")

pearson_r, pearson_p = pearsonr(cos_sim, llm_scores)
spearman_r, spearman_p = spearmanr(cos_sim, llm_scores)

print(f"Pearson r   = {pearson_r:.3f}, p = {pearson_p:.2e}")
print(f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.2e}")

# boxplot data grouped by LLM score
unique_scores = sorted(np.unique(llm_scores))
data = [cos_sim[llm_scores == s] for s in unique_scores]

plt.figure(figsize=(8, 5))
plt.boxplot(data, labels=[str(s) for s in unique_scores], showfliers=True)
plt.xlabel("LLM Score")
plt.ylabel("Cosine Similarity (query · doc)")
# plt.title("Cosine similarity distribution by LLM score")
plt.title(f"Cosine similarity by LLM score (Pearson r={pearson_r:.3f})")
plt.tight_layout()
plt.savefig("results/plot/cos_sim_box_by_llm_score.png", dpi=200)
plt.show()
plt.savefig("results/plot/LLM_score_vs_cosine_similarity_box_plot.png")






breakpoint()