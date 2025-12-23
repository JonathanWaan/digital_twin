import torch
from sentence_transformers import SentenceTransformer

model_name = "myrkur/sentence-transformer-parsbert-fa"
model_name = "Qwen/Qwen3-Embedding-8B"
model = SentenceTransformer(model_name)

emb_norm = model.encode("We may call the faculty of cognition from principles a priori, pure Reason, and the inquiry into its possibility and bounds generally the Critique of pure Reason, although by this faculty we only understand Reason in its theoretical employment, as it appears under that name in the former work; without wishing to inquire into its faculty, as practical Reason, according to its special principles.",normalize_embeddings=True)
emb_no_norm = model.encode("We may call the faculty of cognition from principles a priori, pure Reason, and the inquiry into its possibility and bounds generally the Critique of pure Reason, although by this faculty we only understand Reason in its theoretical employment, as it appears under that name in the former work; without wishing to inquire into its faculty, as practical Reason, according to its special principles.")
breakpoint()
print(emb.shape)

breakpoint()

from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyCaeVOutf0Eaa7-cwaJSK5AW-CysuO5j74")
response = client.models.embed_content(
                model="gemini-embedding-001",
                contents="hello world",
                config=types.EmbedContentConfig(output_dimensionality=768),
            )

breakpoint()

