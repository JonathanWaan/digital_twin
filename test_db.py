from util import cosine_sim_matrix, embed_query
import pandas as pd
from google import genai
from google.genai import types
import torch
from tqdm import tqdm
retrieval = pd.read_csv("data/test_retrieval.csv")
client = genai.Client(api_key="AIzaSyCaeVOutf0Eaa7-cwaJSK5AW-CysuO5j74")

# all_sentences = retrieval['topic'].tolist()
# all_emb = []
# # add tmpq bar

# for sentence in all_sentences:
#     emb = embed_query(sentence, client)
#     all_emb.append(torch.tensor(emb))

# torch.save(torch.stack(all_emb), "data/retrieval_embeddings_gemini.pt")

breakpoint()