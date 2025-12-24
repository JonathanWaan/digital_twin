import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import torch
from pydantic import BaseModel, Field
from typing import Literal

class RelevanceScore(BaseModel):
    justification: str = Field(
        description="Concise justification citing specific words or ideas from the sentence."
    )
    score: int = Field(
        description="Relevance score from 1 (irrelevant) to 5 (most relevant)",
        ge=1,
        le=5,
    )


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


from google.genai import types

score_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "justification": types.Schema(
            type=types.Type.STRING,
            description=("Concise justification (1–2 sentences) citing specific words or ideas "
        "from the sentence that support the assigned score. ")
        ),
        "score": types.Schema(
            type=types.Type.INTEGER,
            description="Relevance score from 1 (irrelevant) to 5 (most relevant)"
        ),
    },
    required=[ "justification", "score"]
)

def build_prompt(query: str, sentence: str) -> str:
    return f"""
You are evaluating *semantic closeness* between a query and a sentence.

Goal: check whether embedding cosine similarity "makes sense" qualitatively.
So score based on topic overlap and conceptual proximity, NOT on whether the
sentence fully answers the question or gives a complete argument.

INPUT QUERY:
"{query}"

SENTENCE:
"{sentence}"

KEY IDEA FOR THIS QUERY:
The core topic is the moral status of LYING / FALSEHOOD / DECEPTION / TRUTHFULNESS,
including whether it is wrong, permissible, excusable, or justified (e.g., by compassion,
utility, necessity, social benefit).

TREAT THESE AS LYING-RELATED SYNONYMS:
lie, lying, falsehood (falshood), deception, deceit, dissimulation, fiction (when contrasted with truth),
veracity, honesty (when about truth-telling), making others believe something false.

SCORING RULES (semantic closeness):
- Score 5 (Extremely close): The sentence is explicitly about lying/falsehood/deception/truth-telling
  AND discusses its moral status (wrong/immoral/virtue/vicious/blame/praise/permissible/justifiable/excusable),
  OR explicitly discusses exceptions/justifications (compassion, utility, necessity, social good, entertainment, etc.).

- Score 4 (Close): The sentence is clearly about lying/falsehood/deception/truthfulness in a substantial way
  (including descriptive, psychological, social, or practical discussion), but does NOT clearly evaluate its moral
  status or does so only indirectly/briefly.

- Score 3 (Moderately related): The sentence is mainly about truth/falsehood in an epistemic or abstract sense
  (e.g., what truth/falsehood is, belief, propositions) OR about morality/virtue in general,
  with only a weak/implicit tie to lying/deception.

- Score 2 (Weakly related): Broadly about morality, rules, compassion, utility, justice, promises, society, etc.
  but not specifically about lying/falsehood/deception/truthfulness.

- Score 1 (Unrelated): Not meaningfully connected.

INSTRUCTIONS:
- Choose the highest score the sentence JUSTIFIES from the sentence alone.
- Do not assume surrounding context.
- If unsure between two scores, choose the LOWER one.
""".strip()



# def build_prompt(query: str, sentence: str) -> str:
#     return f"""
# You are evaluating the relevance of a sentence to a philosophical query.

# INPUT QUERY:
# "{query}"

# SENTENCE:
# "{sentence}"

# SCORING RULES:
# - Score 5: Directly addresses lying, deception, or truthfulness AND evaluates whether it is always wrong or can be justified by outcomes, compassion, necessity, or utility.
# - Score 4: Does not mention lying explicitly, but clearly discusses exceptions to moral rules, necessity, compassion, or social utility that reasonably apply to lying.
# - Score 3: Provides general moral theory (e.g., sentiment, sympathy, utility) without addressing lying or moral exceptions.
# - Score 2: Only weakly or tangentially related (e.g., truth, belief, morality without evaluation).
# - Score 1: Completely irrelevant.

# INSTRUCTIONS:
# - Choose the highest score the sentence JUSTIFIES.
# - If unsure between two scores, choose the LOWER one.
# - Do not assume any context beyond the sentence itself.
# """.strip()

