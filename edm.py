import streamlit as st
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight, GC
from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import io
from scipy.cluster.hierarchy import linkage, dendrogram
import openai

# -----------------------------
# OpenAI API Key
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌐 AI-Powered eDNA Research Platform",
    layout="wide",
)
st.title("🧬 AI-Powered eDNA Platform for Biodiversity Assessment")

# -----------------------------
# Sidebar: File Upload
# -----------------------------
st.sidebar.header("Upload FASTA/FA Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload your sequences (DNA, RNA, Protein)", type=["fasta", "fa"], accept_multiple_files=True
)

# -----------------------------
# Parse sequences
# -----------------------------
def parse_sequences(file):
    try:
        fasta_text = io.StringIO(file.getvalue().decode("utf-8"))
        records = list(SeqIO.parse(fasta_text, "fasta"))
        data = []
        for rec in records:
            seq = rec.seq
            seq_type = 'Protein' if rec.seq.alphabet.__class__.__name__ == 'ProteinAlphabet' else 'DNA/RNA'
            data.append({
                "ID": rec.id,
                "Description": rec.description,
                "Sequence": str(seq),
                "Length": len(seq),
                "GC_Content": GC(seq) if seq_type == 'DNA/RNA' else np.nan,
                "Seq_Type": seq_type,
                "Molecular_Weight": molecular_weight(seq),
                "Reverse_Complement": str(seq.reverse_complement()) if seq_type == 'DNA/RNA' else '-'
            })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error parsing {file.name}: {e}")
        return pd.DataFrame()

# -----------------------------
# Combine all uploaded files
# -----------------------------
dfs = [parse_sequences(f) for f in uploaded_files] if uploaded_files else []
df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if df.empty:
    st.warning("Upload at least one sequence to start analysis.")
else:
    st.subheader(f"Dataset Preview ({len(df)} sequences)")
    st.dataframe(df.head())

# -----------------------------
# Tabs: Stats, AI, BLAST, Tree
# -----------------------------
if not df.empty:
    tabs = st.tabs(["📊 Stats", "💬 AI Chatbot", "🔬 BLAST", "🌳 Phylogenetic Tree", "📈 Biodiversity"])

    # -----------------------------
    # Stats Tab
    # -----------------------------
    with tabs[0]:
        st.subheader("Sequence Statistics")
        st.write(df.describe(include='all'))

    # -----------------------------
    # AI Chatbot Tab
    # -----------------------------
    with tabs[1]:
        st.subheader("AI Chatbot")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        @st.cache_resource
        def embed_sequences(sequences):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(sequences, show_progress_bar=True)

        embeddings = embed_sequences(df['Sequence'].tolist())
        user_question = st.text_input("Ask questions about your sequences")
        if user_question:
            def ai_chat(question):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                q_emb = model.encode([question])
                sims = cosine_similarity(q_emb, embeddings)[0]
                top_idx = np.argsort(sims)[-5:][::-1]
                context = "\n".join([df['Sequence'].iloc[i] for i in top_idx])
                prompt = f"""
You are an AI assistant for eDNA.
Use only the following sequences:
{context}

Previous chats: {st.session_state['chat_history']}
Question: {question}
"""
                try:
                    response = openai.chat.
