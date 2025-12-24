import torch
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from collections import Counter
import math
load_dotenv()



def entropy_from_counts(counts):
    total = sum(counts.values())
    if total == 0:
        return np.nan
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def overlap_rate(curr_idxs, past_idxs):
    if not past_idxs:
        return 0.0
    return len(set(curr_idxs) & set(past_idxs)) / len(curr_idxs)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def build_retrieval_query(
    query_type: str,
    topic: str,
    opponent_msg: str,
    history: list,
    last_self_msg: str | None,
):
    if query_type == "Q1":
        return f"{topic}\n{opponent_msg}"

    elif query_type == "Q2":
        extra = last_self_msg if last_self_msg else ""
        return f"{topic}\n{opponent_msg}\n{extra}"

    elif query_type == "Q3":
        return f"{topic}\n" + "\n".join(history)

    else:
        raise ValueError(f"Unknown query_type: {query_type}")


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
    # if author == "Kant":
    #     author += "s"
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
    topic: str,
    query_type: str,
    metrics_log: list,
    k: int = 5,
):
    # ----- build retrieval query -----
    last_self_msg = None
    if history and history[-1].startswith(author):
        last_self_msg = history[-1]

    retrieval_query = build_retrieval_query(
        query_type=query_type,
        topic=topic,
        opponent_msg=opponent_last_msg,
        history=history,
        last_self_msg=last_self_msg,
    )

    # ----- embed query -----
    query_emb = embed_query(retrieval_query)
    
    


    # ----- retrieve -----
    retrieved, idxs = retrieve_sentences(retrieval_query, author, k)

    # ----- metrics bookkeeping -----
    author_state = retrieval_state[author]

    if author_state["anchor_query_emb"] is None:
        author_state["anchor_query_emb"] = query_emb

    # overlap %
    overlap = overlap_rate(idxs, author_state["all_idxs"])

    # entropy
    for i in idxs:
        author_state["idx_counter"][int(i)] += 1
    ent = entropy_from_counts(author_state["idx_counter"])

    # query-query cosine
    qq_cos = (
        cosine(query_emb, author_state["last_query_emb"])
        if author_state["last_query_emb"] is not None
        else np.nan
    )
    qq_anchor = cosine(query_emb, author_state["anchor_query_emb"])
    # log metrics
    metrics_log.append({
        "author": author,
        "round": author_state["round"],
        "query_type": query_type,
        "entropy": ent,
        "overlap_pct": overlap,
        "query_query_cos": qq_cos,
        "query_anchor_cos": qq_anchor,
    })

    # update state
    author_state["all_idxs"].extend(idxs)
    author_state["last_query_emb"] = query_emb
    author_state["round"] += 1
    db_history_author[author].append(idxs)

    # ----- generate response -----
    prompt = build_prompt(
        author=author,
        retrieved_df=retrieved,
        opponent_msg=opponent_last_msg,
        history=history,
    )

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
    )

    return response.text.strip()

        
        
        
# def philosopher_reply(
#     author: str,
#     opponent_last_msg: str,
#     history: list,
#     k: int = 5,
# ):
#     """
#     Single turn reply by one philosopher agent.
#     """
#     if not history:
#         retrieved, idxs = retrieve_sentences(opponent_last_msg, author, k)
#     else:
#         retrieved, idxs = retrieve_sentences("".join(history), author, k)
#     db_history_author[author].append(idxs.tolist())

#     prompt = build_prompt(
#         author=author,
#         retrieved_df=retrieved,
#         opponent_msg=opponent_last_msg,
#         history=history,
#     )


#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt,
#     )

#     return response.text.strip()


def debate_two_philosophers(
    author_a: str,
    author_b: str,
    topic: str,
    query_type: str,
    n_rounds: int = 5,
    k: int = 5,
):

    history = []

    print(f"\n🧠 Debate initialized ({query_type})")
    print(f"{author_a} vs {author_b}")
    print(f"Topic: {topic}")
    print("=" * 60)

    a_msg = philosopher_reply(
        author=author_a,
        opponent_last_msg=topic,
        history=history,
        topic=topic,
        query_type=query_type,
        metrics_log=metrics_log,
        k=k,
    )
    history.append(f"{author_a}: {a_msg}")
    print(author_a+": "+a_msg+"\n")
    
    for rnd in range(n_rounds):
        print("current round", rnd)
        b_msg = philosopher_reply(
            author=author_b,
            opponent_last_msg=a_msg,
            history=history,
            topic=topic,
            query_type=query_type,
            metrics_log=metrics_log,
            k=k,
        )
        history.append(f"{author_b}: {b_msg}")
        print(author_b+": "+b_msg+"\n")
        a_msg = philosopher_reply(
            author=author_a,
            opponent_last_msg=b_msg,
            history=history,
            topic=topic,
            query_type=query_type,
            metrics_log=metrics_log,
            k=k,
        )
        history.append(f"{author_a}: {a_msg}")
        print(author_a+": "+a_msg+"\n")
        print(db_history_author)
        
        
        print(f"--- Round finished ---")

    return history


# def debate_two_philosophers(
#     author_a: str,
#     author_b: str,
#     topic: str,
#     n_rounds: int = 5,
#     k: int = 5,
# ):
#     history = []

#     print(f"\n🧠 Debate initialized")
#     print(f"🅰️ {author_a}  vs  🅱️ {author_b}")
#     print(f"📌 Topic: {topic}")
#     print("=" * 60)

#     # Initial statement by A
#     a_msg = philosopher_reply(
#         author=author_a,
#         opponent_last_msg=topic,
#         history=history,
#         k=k,
#     )
#     print(f"\n{author_a}: {a_msg}\n")
#     history.append(f"{author_a}: {a_msg}")

#     for r in range(1, n_rounds + 1):
#         print(f"--- Round {r} ---")

#         b_msg = philosopher_reply(
#             author=author_b,
#             opponent_last_msg=a_msg,
#             history=history,
#             k=k,
#         )
#         print(f"\n{author_b}: {b_msg}\n")
#         history.append(f"{author_b}: {b_msg}")

#         a_msg = philosopher_reply(
#             author=author_a,
#             opponent_last_msg=b_msg,
#             history=history,
#             k=k,
#         )
#         print(f"\n{author_a}: {a_msg}\n")
#         history.append(f"{author_a}: {a_msg}")

#     print("=" * 60)
#     print("🏁 Debate finished.")

#     # ===== Judge decision =====
#     print("\n⚖️ Judge deliberating...\n")

#     judgment = judge_debate(
#         history=history,
#         author_a=author_a,
#         author_b=author_b,
#     )

#     print("🧑‍⚖️ Judge Verdict:")
#     print(judgment)

#     return history, judgment

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
retrieval_df = pd.read_csv("data/test_retrieval.csv")

assert embeddings.shape[0] == len(sentences_df)

# Convert to NumPy once for fast retrieval
embeddings_np = embeddings.numpy()

# ===== Gemini client =====
client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
author_a = "Kant"
author_b = "Hume"
# philosopher_chat(author="Kant", k=6)

retrieval_state = {
    author_a: {
        "idx_counter": Counter(),
        "all_idxs": [],
        "last_query_emb": None,
        "anchor_query_emb":None,
        "round": 0,
    },
    author_b: {
        "idx_counter": Counter(),
        "all_idxs": [],
        "last_query_emb": None,
        "anchor_query_emb":None,
        "round": 0,
    },
}

metrics_log = []


db_history_author = {author_a:[], author_b:[]}


for qtype in ["Q1", "Q2", "Q3"]:
    retrieval_state = {
        author_a: {"idx_counter": Counter(), "all_idxs": [], "last_query_emb": None, "anchor_query_emb":None,"round": 0},
        author_b: {"idx_counter": Counter(), "all_idxs": [], "last_query_emb": None, "anchor_query_emb":None,"round": 0},
    }
    metrics_log = []

    debate_two_philosophers(
        author_a=author_a,
        author_b=author_b,
        topic="Where does morality come from: reason or feelings?",
        query_type=qtype,
        n_rounds=10,
        k=6,
    )

    df = pd.DataFrame(metrics_log)
    df.to_csv(f"metrics_{qtype}.csv", index=False)
    breakpoint()



# debate_two_philosophers(
#     author_a=author_a,
#     author_b=author_b,
#     topic="Where does morality come from: reason or feelings?",
#     n_rounds=4,
#     k=6,
# )