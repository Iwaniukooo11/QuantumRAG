import streamlit as st
import json
from context_retriever import ContextRetriever
#from grover import grover_top_k
#from gpt_final_answer import generate_answer_from_contexts

MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K_FROM_FAISS = 10
TOP_K_FINAL = 3

EMBEDDINGS_FILE = "saved_embeddings/squad_embeddings_mixedbread_ai_mxbai_embed_large_v1.npy"
DOCS_LIST_FILE = "saved_embeddings/squad_docs_mixedbread_ai_mxbai_embed_large_v1.json"

# Context retriever
retriever = ContextRetriever(
    model_name=MODEL_NAME,
    embeddings_file=EMBEDDINGS_FILE,
    docs_list_file=DOCS_LIST_FILE
)

with open(DOCS_LIST_FILE, 'r', encoding='utf-8') as f:
    docs = json.load(f)
retriever.build_index(docs)

# Streamlit UI
st.title("QA with Grover and GPT")
st.markdown("Ask a question and get an answer using Grover and GPT.")

user_question = st.text_input("Your question:", placeholder="Type your question here...")

if user_question:
    query = QUERY_PREFIX + user_question
    with st.spinner("Searching for relevant contexts..."):
        # Use the retriever to get the top 10 results
        top_10_results = retriever.search(query, top_k=TOP_K_FROM_FAISS)

    if not top_10_results:
        st.error("No relevant contexts found.")
    else:
        with st.spinner("Searching for relevant contexts with Grover..."):
            # Use Grover to get the top k contexts
            #top_k_contexts = grover_top_k_simulation(top_10_results, k=TOP_K_FINAL)
            top_k_contexts = ["TODO Ania: implement this function to get top k contexts from Grover"]

        with st.spinner("Generating answer..."):
            # Generate the final answer from the top k contexts
            #final_answer = generate_answer_from_contexts(user_question, top_k_contexts)
            final_answer = "TODO Mateusz: implement this function to get final answer from gpt"

        st.subheader("Answer:")
        if final_answer is None:
            st.error("No answer found.")
        else:
            st.success("Answer found!")
        st.write(final_answer)

        with st.expander("Show top 3 contexts"):
            st.write(top_k_contexts)
            #for i, ctx in enumerate(top_k_contexts):
                #st.markdown(f"**Context {i+1} (score: {ctx['similarity_score']:.4f})**")
                #st.write(ctx["document"])

        with st.expander("Show all top 10 contexts"):
            for res in top_10_results:
                st.markdown(f"**Rank {res['rank']} (score: {res['similarity_score']:.4f})**")
                st.write(res["document"][:500] + "...")
