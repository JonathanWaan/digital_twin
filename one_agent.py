import torch
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
def cosine_sim_matrix(query_emb, doc_embs):
    """
    query_emb: (768,)
    doc_embs: (N, 768)
    returns: (N,)
    """

    query_emb = query_emb / np.linalg.norm(query_emb)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return doc_embs @ query_emb


def embed_query(text: str) -> np.ndarray:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return np.array(response.embeddings[0].values)

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



def build_prompt(author, retrieved_df, opponent_msg, history):
    context = "\n".join(
        f"- {row.sentence}" for _, row in retrieved_df.iterrows()
    )

    conversation = "\n".join(history[-6:])  # limit memory to avoid drift

    prompt = f"""
You are {author}, engaged in a philosophical debate.

You must follow ALL rules strictly.

=== YOUR KNOWLEDGE (authoritative, do not invent beyond this) ===
{context}

=== CONVERSATION SO FAR ===
{conversation}

=== OPPONENT'S LATEST CLAIM ===
"{opponent_msg}"

=== YOUR TASK ===
1. Identify ONE specific philosophical claim made by your opponent.
2. Choose EXACTLY ONE of the following actions:
   - REFUTE it with reasoning grounded in your philosophy
   - QUALIFY it by showing where it partially holds and where it fails
   - CONCEDE it explicitly if it cannot be answered within your framework
3. Advance the debate. DO NOT restate your general philosophy unless it directly targets the opponent’s claim.
4. If you find yourself repeating earlier points, you MUST either:
   - deepen the argument, OR
   - concede that no further progress is possible.

=== ALLOWED END STATES ===
- "I concede this point."
- "This disagreement cannot be resolved within my philosophical framework."

=== STYLE CONSTRAINTS ===
- Speak in the historical style of {author}, but prioritize clarity over ornament.
- Be concise, sharp, and dialectical.
- No repetition, no summaries of your own doctrine.

Begin your response directly.
"""
    return prompt.strip()



# def build_prompt(author, retrieved_df, user_query, history):

#     context = "\n".join(
#         f"- {row.sentence}" for _, row in retrieved_df.iterrows()
#     )

#     conversation = "\n".join(history)

#     prompt = f"""
# You are a philosopher speaking strictly in the style of {author}.

# Here are {author}'s thoughts relevant to the current discussion:
# {context}

# Conversation so far:
# {conversation}

# User question:
# {user_query}

# Debate to the conversation opponent using the retrieved context as your knowledge base.
# """

#     return prompt.strip()

def philosopher_chat(author, k=5):
    history = []

    print(f"🧠 Philosopher RAG system initialized for: {author}")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("User: ")
        if user_query.lower() == "exit":
            break

        retrieved = retrieve_sentences(user_query, author, k)
        prompt = build_prompt(author, retrieved, user_query, history)

        # Generate answer
        response = client.models.generate_content(
            # model="gemini-2.5-flash",
            model = "gemini-3-pro-preview",
            contents=prompt,
        )

        answer = response.text
        print(f"\n{author}: {answer}\n")

        # Update memory
        history.append(f"User: {user_query}")
        history.append(f"{author}: {answer}")
        
def philosopher_reply(
    author: str,
    opponent_last_msg: str,
    history: list,
    k: int = 5,
):
    """
    Single turn reply by one philosopher agent.
    """
    if not history:
        retrieved, idxs = retrieve_sentences(opponent_last_msg, author, k)
    else:
        retrieved, idxs = retrieve_sentences("".join(history), author, k)
    db_history_author[author].append(idxs.tolist())
    print(db_history_author)
    prompt = build_prompt(
        author=author,
        retrieved_df=retrieved,
        opponent_msg=opponent_last_msg,
        history=history,
    )


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()

def debate_two_philosophers(
    author_a: str,
    author_b: str,
    topic: str,
    n_rounds: int = 5,
    k: int = 5,
):
    history = []

    print(f"\n🧠 Debate initialized")
    print(f"🅰️ {author_a}  vs  🅱️ {author_b}")
    print(f"📌 Topic: {topic}")
    print("=" * 60)

    # Initial statement by A
    a_msg = philosopher_reply(
        author=author_a,
        opponent_last_msg=topic,
        history=history,
        k=k,
    )
    print(f"\n{author_a}: {a_msg}\n")
    history.append(f"{author_a}: {a_msg}")

    for r in range(1, n_rounds + 1):
        print(f"--- Round {r} ---")

        b_msg = philosopher_reply(
            author=author_b,
            opponent_last_msg=a_msg,
            history=history,
            k=k,
        )
        print(f"\n{author_b}: {b_msg}\n")
        history.append(f"{author_b}: {b_msg}")

        a_msg = philosopher_reply(
            author=author_a,
            opponent_last_msg=b_msg,
            history=history,
            k=k,
        )
        print(f"\n{author_a}: {a_msg}\n")
        history.append(f"{author_a}: {a_msg}")

    print("=" * 60)
    print("🏁 Debate finished.")

    # ===== Judge decision =====
    print("\n⚖️ Judge deliberating...\n")

    judgment = judge_debate(
        history=history,
        author_a=author_a,
        author_b=author_b,
    )

    print("🧑‍⚖️ Judge Verdict:")
    print(judgment)

    return history, judgment

def judge_debate(
    history: list,
    author_a: str,
    author_b: str,
):
    """
    Judge reads the full debate and decides a winner.
    """

    conversation = "\n".join(history)

    judge_prompt = f"""
You are a neutral philosophy judge.

Below is a debate between two philosophers:
- {author_a}
- {author_b}

Your task:
1. Decide which philosopher presented the stronger argument.
2. Justify your decision concisely.
3. Base your judgment on clarity, logical consistency, depth, and faithfulness to their philosophical framework.

Debate transcript:
{conversation}

Return your answer in the following format:

Winner: <{author_a} or {author_b}>
Reasoning: <short explanation>
"""

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=judge_prompt,
    )

    return response.text.strip()

# ===== Load data =====
sentences_df = pd.read_csv("data/sentences.csv")
embeddings = torch.load("data/gemini_768_sentence.pt")  # shape: (N, 768)

assert embeddings.shape[0] == len(sentences_df)

# Convert to NumPy once for fast retrieval
embeddings_np = embeddings.numpy()

# ===== Gemini client =====
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
author_a = "Kant"
author_b = "Hume"
# philosopher_chat(author="Kant", k=6)

db_history_author = {author_a:[], author_b:[]}
debate_two_philosophers(
    author_a=author_a,
    author_b=author_b,
    topic="Where does morality come from: reason or feelings?",
    n_rounds=4,
    k=6,
)