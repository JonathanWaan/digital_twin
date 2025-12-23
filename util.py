import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import torch


def cosine_sim_matrix(query_emb, doc_embs):
    """
    query_emb: (768,)
    doc_embs: (N, 768)
    returns: (N,)
    """

    query_emb = query_emb / np.linalg.norm(query_emb)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return doc_embs @ query_emb

def embed_query(text: str, client) -> np.ndarray:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return np.array(response.embeddings[0].values)
