import torch
from google import genai
from google.genai import types
from util import build_prompt, score_schema, RelevanceScore
import os
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

def score_sentence(client, model: str, query: str, sentence: str):
    prompt = build_prompt(query, sentence)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": RelevanceScore.model_json_schema(),
            "temperature": 0.0,
            "max_output_tokens": 256,
        },
    )

    # Parse + validate
    return RelevanceScore.model_validate_json(response.text)




LLM_label_package = torch.load('results/experiment_results/LLM_label_package_1_numpy_topk.pt',weights_only=False)

retrieved_sentence = LLM_label_package['retrieved_sentence']['sentence'].tolist()
query = LLM_label_package['query']
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))


MODEL_NAME = "gemini-2.0-flash"  # or gemini-2.0-flash if you want speed

results = []

for sent in tqdm(retrieved_sentence):
    try:
        out = score_sentence(
            client=client,
            model=MODEL_NAME,
            query=query,
            sentence=sent,
        )
        
        results.append({
            "sentence": sent,
            "score": out.score,
            "justification": out.justification,
        })

    except Exception as e:
        results.append({
            "sentence": sent,
            "score": None,
            "justification": f"ERROR: {e}",
        })

try:
    torch.save(results, 'results/experiment_results/llm_scored_sentences_topical_relavence.pt')
except:
    pass
[res["score"] for res in results]
breakpoint()


breakpoint()