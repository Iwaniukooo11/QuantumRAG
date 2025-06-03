import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.components.AgentHandler import AgentHandler
from src.components.ContextRetriever import ContextRetriever
from src.components.GroverTopK import GroverTopK
from src.utils.ContextRetrieverUtils import ContextRetrieverUtils

# parametry testu
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
MODEL_KEYS = ["llama-3-8b", "mixtral-8x7b", "phi-3.5"]
EMBEDDINGS_FILE = "saved_embeddings/squad_embeddings_mixedbread_ai_mxbai_embed_large_v1.npy"
DOCS_LIST_FILE = "saved_embeddings/squad_docs_mixedbread_ai_mxbai_embed_large_v1.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K_VARIANTS = {
    "grover_top3": (True, 3),
    "classic_top3": (False, 3),
    "classic_top1": (False, 1),
    "grover_top1": (True, 1),
    "no_context": (False, 0),
}

# dane: pytania i idealne opowiedzi z SQuAD 1.1
squad_path = "squad_dataset/train-v1.1.json"
with open(DOCS_LIST_FILE, "r", encoding="utf-8") as f:
    indexed_documents = json.load(f)

with open(squad_path, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

samples = []
count_all = 0
for article in squad_data["data"]:
    count_art = 0
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            if qa["answers"]:
                samples.append({
                    "question": qa["question"],
                    "ideal_answer": qa["answers"][0]["text"]
                })
                count_art += 1
                count_all += 1
                if count_art >= 2:
                    break
    if count_all >= 10:
        break

print(f"Loaded {len(samples)} QA pairs from SQuAD 1.1")


retriever = ContextRetriever(MODEL_NAME, EMBEDDINGS_FILE, DOCS_LIST_FILE)
retriever.build_index(indexed_documents)
agent_handler = AgentHandler(model_keys=MODEL_KEYS)
grover = GroverTopK()

def word_overlap(a, b):
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    return round(len(a_tokens & b_tokens) / len(b_tokens) * 100, 2) if b_tokens else 0.0

def embed(text):
    model = retriever.model
    tokenizer = retriever.tokenizer
    device = retriever.device
    emb = ContextRetrieverUtils.get_embeddings([text], model, tokenizer, device)
    return emb

# testy
results = []
for idx, sample in enumerate(tqdm(samples, desc="Running tests")):
    question = sample["question"]
    ideal = sample["ideal_answer"]
    query = QUERY_PREFIX + question
    top10_context_selection_time = time.time()

    top10 = retriever.search(query, top_k=10)
    top10_context_selection_time = round(time.time() - top10_context_selection_time, 2)

    for variant_name, (use_grover, top_k) in TOP_K_VARIANTS.items():
        if top_k > 0:
            start_context_top3 = time.time()
            if use_grover:
                selected = grover.select(top10, k=top_k)
                contexts = selected["contexts"]
            else:
                contexts = top10[:top_k]
            top3_context_selection_time = round(time.time() - start_context_top3, 2)
        else:
            contexts = []
            top3_context_selection_time = 0.0

        for model_key in MODEL_KEYS:
            start_gen = time.time()
            response = agent_handler.generate(question, contexts if top_k > 0 else None, model_key)
            generation_time = round(time.time() - start_gen, 2)

            # podobnie jak wcześniej dodaj do results
            results.append({
                "sample_id": idx,
                "question": question,
                "ideal_answer": ideal,
                "model": model_key,
                "variant": variant_name,
                "predicted": response,
                "cosine": round(cosine_similarity(embed(ideal), embed(response))[0][0], 4),
                "overlap": word_overlap(ideal, response),
                "top10_contexts_selection_time_s": top10_context_selection_time,
                "top_k_context_selection_time_s(grover_or_classic)": top3_context_selection_time,
                "generation_time_s": generation_time,
                "contexts_used": [c["document"] if isinstance(c, dict) else str(c) for c in contexts]
            })

# zapis wyników do pliku
df = pd.DataFrame(results)
df.to_json("tests/time_test_results.json", indent=2, force_ascii=False)
df.to_csv("tests/time_test_result.csv", index=False)
print("Testy zakończone. Wyniki zapisane.")