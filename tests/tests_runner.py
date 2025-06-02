import os
import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                if count_art >= 10:
                    break
    if count_all >= 50:
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
    top10 = retriever.search(query, top_k=10)

    for variant_name, (use_grover, top_k) in TOP_K_VARIANTS.items():
        if top_k > 0:
            if use_grover:
                contexts = grover.select(top10, k=top_k)["contexts"]
            else:
                contexts = top10[:top_k]
        else:
            contexts = []

        start_time = time.time()
        answers_dict = agent_handler.compare_all(question, contexts)
        elapsed_time = round(time.time() - start_time, 2)

        for model_key, outputs in answers_dict.items():
            pred_with = outputs["with_context"]
            pred_without = outputs["without_context"]

            overlap = word_overlap(pred_with, pred_without)
            ideal_emb = embed(ideal)
            with_emb = embed(pred_with)
            without_emb = embed(pred_without)
            cos_with = round(cosine_similarity(ideal_emb, with_emb)[0][0], 4)
            cos_without = round(cosine_similarity(ideal_emb, without_emb)[0][0], 4)

            results.append({
                "sample_id": idx,
                "question": question,
                "ideal_answer": ideal,
                "model": model_key,
                "variant": variant_name,
                "word_overlap": overlap,
                "cosine_with": cos_with,
                "cosine_without": cos_without,
                "pred_with_context": pred_with,
                "pred_without_context": pred_without,
                "ideal_tokens": set(ideal.lower().split()),
                "with_tokens": set(pred_with.lower().split()),
                "without_tokens": set(pred_without.lower().split()),
                "total_variant_runtime_s": elapsed_time
            })

# zapis wyników do pliku
df = pd.DataFrame(results)
df.to_json("tests/test_results.json", indent=2, force_ascii=False)
df.to_csv("tests/test_results.csv", index=False)
print("Testy zakończone. Wyniki zapisane.")