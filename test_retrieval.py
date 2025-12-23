import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import torch
import time
from itertools import combinations
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

def mean_query_cosine_similarity(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    idx: np.ndarray,
):
    # cosine similarity since both are normalized
    sims = doc_embs[idx] @ query_emb
    return float(np.mean(sims))





def build_faiss_indices(doc_embs: np.ndarray):
    """
    doc_embs: (N, d), assumed L2-normalized for cosine/IP
    """
    N, d = doc_embs.shape

    # ---- FAISS Flat (exact) ----
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(doc_embs.astype(np.float32))

    # ---- FAISS HNSW (ANN) ----
    index_hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(doc_embs.astype(np.float32))

    # ---- FAISS IVF (ANN) ----
    nlist = min(100, int(np.sqrt(N)))  # safe heuristic
    quantizer = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(
        quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
    )
    index_ivf.train(doc_embs.astype(np.float32))
    index_ivf.nprobe = min(10, nlist)
    index_ivf.add(doc_embs.astype(np.float32))

    return {
        "faiss_flat": index_flat,
        "faiss_hnsw": index_hnsw,
        "faiss_ivf": index_ivf,
    }

def mean_pairwise_cosine_distance(embs: np.ndarray):
    """
    embs: (k, d)
    """
    if len(embs) <= 1:
        return 0.0

    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    dists = []

    for i, j in combinations(range(len(embs)), 2):
        cos_sim = np.dot(embs[i], embs[j])
        dists.append(1 - cos_sim)

    return float(np.mean(dists))

def numpy_topk(query_emb, doc_embs, k):
    sims = doc_embs @ query_emb
    return np.argsort(sims)[-k:][::-1]


# def retrieve_method(
#     method_name: str,
#     query_emb: np.ndarray,
#     doc_embs: np.ndarray,
#     k: int,
# ):
#     start = time.perf_counter()

#     if method_name == "numpy_topk":
#         idx = numpy_topk(query_emb, doc_embs, k)

#     elif method_name == "numpy_topk_largeM":
#         # simulate candidate-pool expansion (M > k)
#         M = 50
#         sims = doc_embs @ query_emb
#         cand_idx = np.argsort(sims)[-M:]
#         idx = cand_idx[np.argsort(sims[cand_idx])[-k:]][::-1]

#     elif method_name == "random_baseline":
#         idx = np.random.choice(len(doc_embs), size=k, replace=False)

#     elif method_name == "topk_no_norm":
#         sims = doc_embs @ query_emb  # intentionally skip normalization
#         idx = np.argsort(sims)[-k:][::-1]

#     elif method_name == "shuffled_topk":
#         idx = numpy_topk(query_emb, doc_embs, k)
#         np.random.shuffle(idx)

#     else:
#         raise ValueError(method_name)

#     latency_ms = (time.perf_counter() - start) * 1000
#     return idx, latency_ms




def retrieve_method(
    method_name: str,
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    k: int,
    faiss_indices=None,
):
    start = time.perf_counter()

    if method_name == "numpy_topk":
        idx = numpy_topk(query_emb, doc_embs, k)

    elif method_name == "faiss_flat":
        D, I = faiss_indices["faiss_flat"].search(
            query_emb[None, :].astype(np.float32), k
        )
        idx = I[0]

    elif method_name == "faiss_hnsw":
        D, I = faiss_indices["faiss_hnsw"].search(
            query_emb[None, :].astype(np.float32), k
        )
        idx = I[0]

    elif method_name == "faiss_ivf":
        D, I = faiss_indices["faiss_ivf"].search(
            query_emb[None, :].astype(np.float32), k
        )
        idx = I[0]

    elif method_name == "random_baseline":
        idx = np.random.choice(len(doc_embs), size=k, replace=False)

    else:
        raise ValueError(method_name)

    latency_ms = (time.perf_counter() - start) * 1000
    return idx, latency_ms

def run_rq1_experiment(
    author="Hume",
    k=5,
):
    mask = sentences_df["author"] == author
    doc_embs = embeddings_np[mask.values]

    # normalize once (important for cosine/IP)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    # build indices
    faiss_indices = build_faiss_indices(doc_embs)

    methods = [
        "numpy_topk",
        "faiss_flat",
        "faiss_hnsw",
        "faiss_ivf",
        "random_baseline",
    ]

    results = []

    for q_idx, row in retrieval_df.iterrows():
        if q_idx % 10 == 0:
            print(q_idx)
        query_emb = retrieval_embedding[q_idx].numpy()
        query_emb = query_emb / np.linalg.norm(query_emb)

        gold_idx = numpy_topk(query_emb, doc_embs, k)

        for method in methods:
            idx, latency = retrieve_method(
                method,
                query_emb,
                doc_embs,
                k,
                faiss_indices=faiss_indices,
            )


            recall = len(set(idx) & set(gold_idx)) / k
            diversity = mean_pairwise_cosine_distance(doc_embs[idx])
            mean_cos_sim = mean_query_cosine_similarity(
                query_emb,
                doc_embs,
                idx,
            )

            results.append({
                "query_id": q_idx,
                "method": method,
                "latency_ms": latency,
                "recall_at_k": recall,
                "diversity": diversity,
                "mean_query_cos_sim": mean_cos_sim,
            })

    return pd.DataFrame(results)


# def run_rq1_experiment(
#     author="Hume",
#     k=5,
# ):

#     mask = sentences_df["author"] == author
#     doc_embs = embeddings_np[mask.values]

#     methods = [
#         "numpy_topk",
#         "numpy_topk_largeM",
#         "topk_no_norm",
#         "shuffled_topk",
#         "random_baseline",
#     ]

#     results = []

#     for q_idx, row in retrieval_df.iterrows():
#         query_text = row["topic"]
#         query_emb = retrieval_embedding[q_idx].numpy()
#         query_emb = query_emb / np.linalg.norm(query_emb)

#         # gold standard
#         gold_idx = numpy_topk(query_emb, doc_embs, k)

#         for method in methods:
#             idx, latency = retrieve_method(
#                 method,
#                 query_emb,
#                 doc_embs,
#                 k,
#             )

#             recall = len(set(idx) & set(gold_idx)) / k
#             diversity = mean_pairwise_cosine_distance(doc_embs[idx])

#             results.append({
#                 "query_id": q_idx,
#                 "method": method,
#                 "latency_ms": latency,
#                 "recall_at_k": recall,
#                 "diversity": diversity,
#             })

#     return pd.DataFrame(results)

def retrieve_sentences(
    query: str,
    author: str,
    k: int = 5,
):
    if author == "Kant":
        author += "s"
    # Filter by philosopher
    mask = sentences_df["author"] == author
    sub_df = sentences_df[mask].reset_index(drop=True)
    sub_embs = embeddings_np[mask.values]

    # Embed query
    query_emb = embed_query(query)

    # Compute cosine similarity
    sims = cosine_sim_matrix(query_emb, sub_embs)

    # Top-k indices
    topk_idx = np.argsort(sims)[-k:][::-1]

    return sub_df.iloc[topk_idx][["sentence", "author"]], topk_idx




sentences_df = pd.read_csv("data/sentences.csv")
embeddings = torch.load("data/gemini_768_sentence.pt")  # shape: (N, 768)
embeddings_np = embeddings.numpy()
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
retrieval_df = pd.read_csv("data/test_retrieval.csv")
retrieval_embedding = torch.load("data/retrieval_embeddings_gemini.pt")

# breakpoint()

AUTHORS = ["Hume", "Kant", "Plato", "Aristotle", "Nietzsche"]
KS = [5, 10, 20, 50, 100,200]
# AUTHORS = ['Hume']
# KS = [200]

all_results = []

for author in AUTHORS:
    print(f"Running RQ1 for author={author}")
    for k in KS:
        print(f"  k={k}")
        df = run_rq1_experiment(author=author, k=k)

        # annotate metadata
        df["author"] = author
        df["k"] = k

        all_results.append(df)

rq1_results = pd.concat(all_results, ignore_index=True)

# rq1_results.to_csv("results/experiment_results/rq1_results_diff_k_author_cos_sim.csv", index=False)

breakpoint()

# retrieve_sentences("hello world", "Hume", 5)

