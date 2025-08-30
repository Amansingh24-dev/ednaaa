import streamlit as st
from Bio import SeqIO
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# -----------------------------
# OpenAI API Key
# -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI eDNA FASTA Analysis", layout="wide")
st.title("🧬 AI-Powered eDNA FASTA Analysis Platform")

# -----------------------------
# File Upload
# -----------------------------
st.sidebar.header("Upload FASTA files")
uploaded_files = st.sidebar.file_uploader(
    "Upload your FASTA sequences", type=["fasta", "fa"], accept_multiple_files=True
)

# -----------------------------
# Parse FASTA
# -----------------------------
import io

def parse_fasta(uploaded_file):
    # Convert uploaded bytes file to text
    fasta_text = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    records = list(SeqIO.parse(fasta_text, "fasta"))
    
    data = {
        "id": [rec.id for rec in records],
        "description": [rec.description for rec in records],
        "sequence": [str(rec.seq) for rec in records],
        "length": [len(rec.seq) for rec in records],
        "GC_content": [100 * (rec.seq.count("G") + rec.seq.count("C")) / len(rec.seq) for rec in records]
    }
    return pd.DataFrame(data)


dfs = []
if uploaded_files:
    for f in uploaded_files:
        dfs.append(parse_fasta(f))
    df = pd.concat(dfs, ignore_index=True)
    st.subheader(f"Combined FASTA Dataset Preview ({len(df)} sequences)")
    st.dataframe(df.head())
else:
    st.warning("Please upload at least one FASTA file.")

# -----------------------------
# Sequence Filters
# -----------------------------
st.sidebar.subheader("Filters")
if uploaded_files and not df.empty:
    min_len = int(df['length'].min())
    max_len = int(df['length'].max())
    
    if min_len == max_len:
        st.sidebar.write(f"All sequences have length: {min_len}")
        length_filter = (min_len, max_len)
    else:
        length_filter = st.sidebar.slider(
            "Sequence length range", 
            min_value=min_len, 
            max_value=max_len, 
            value=(min_len, max_len)
        )
        df = df[df['length'].between(*length_filter)]

# -----------------------------
# Quick Stats
# -----------------------------
if uploaded_files:
    st.subheader("📊 Sequence Statistics")
    st.write(f"Total sequences: {len(df)}")
    st.write(f"Average length: {df['length'].mean():.2f}")
    st.write(f"Average GC content: {df['GC_content'].mean():.2f}%")

    st.subheader("Top 20 Longest Sequences")
    st.bar_chart(df.sort_values("length", ascending=False).head(20)[["length"]])

# -----------------------------
# Embeddings for AI Chatbot
# -----------------------------
@st.cache_resource
def embed_sequences(sequences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sequences, show_progress_bar=True)
    return embeddings

if uploaded_files:
    embeddings = embed_sequences(df['sequence'].tolist())

# -----------------------------
# Persistent AI Memory
# -----------------------------
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def ai_chat(question, embeddings, sequences, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([question])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    context = "\n".join([sequences[i] for i in top_idx])
    
    prompt = f"""
    You are an intelligent eDNA assistant.
    Use ONLY the context below to answer questions.
    CONTEXT:
    {context}

    Previous chats: {st.session_state['chat_history']}

    QUESTION:
    {question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content
    st.session_state['chat_history'].append({"Q": question, "A": answer})
    return answer

st.subheader("💬 AI Chatbot for eDNA Sequences")
user_question = st.text_input("Ask the AI about your sequences")
if uploaded_files and user_question:
    answer = ai_chat(user_question, embeddings, df['sequence'].tolist())
    st.write("**AI Answer:**", answer)

# -----------------------------
# Downloadable Results
# -----------------------------
if uploaded_files:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset CSV",
        data=csv,
        file_name='edna_fasta_analysis.csv',
        mime='text/csv',
    )
