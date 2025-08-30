import streamlit as st
from Bio import SeqIO
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import io

# -----------------------------
# OpenAI API Key
# -----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌐 AI-Powered eDNA Worldwide Analysis",
    layout="wide",
)
st.title("🧬 AI-Powered eDNA FASTA Analysis Platform")

# -----------------------------
# Sidebar - File Upload
# -----------------------------
st.sidebar.header("Upload FASTA Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload your FASTA sequences (FASTA/FA)",
    type=["fasta", "fa"],
    accept_multiple_files=True
)

# -----------------------------
# Parse FASTA
# -----------------------------
def parse_fasta(file):
    try:
        fasta_text = io.StringIO(file.getvalue().decode("utf-8"))
        records = list(SeqIO.parse(fasta_text, "fasta"))
        if not records:
            return pd.DataFrame()
        data = {
            "ID": [rec.id for rec in records],
            "Description": [rec.description for rec in records],
            "Sequence": [str(rec.seq) for rec in records],
            "Length": [len(rec.seq) for rec in records],
            "GC_Content": [100 * (rec.seq.count("G") + rec.seq.count("C")) / len(rec.seq) for rec in records]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error parsing {file.name}: {e}")
        return pd.DataFrame()

# -----------------------------
# Load & Combine Data
# -----------------------------
dfs = [parse_fasta(f) for f in uploaded_files] if uploaded_files else []
df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if df.empty:
    st.warning("Upload at least one valid FASTA file to begin analysis.")
else:
    st.subheader(f"Combined Dataset Preview ({len(df)} sequences)")
    st.dataframe(df.head())

# -----------------------------
# Filters
# -----------------------------
if not df.empty:
    st.sidebar.subheader("Sequence Filters")
    min_len, max_len = int(df['Length'].min()), int(df['Length'].max())
    if min_len != max_len:
        length_range = st.sidebar.slider(
            "Sequence length range", min_len, max_len, (min_len, max_len)
        )
        df = df[df['Length'].between(*length_range)]
    st.sidebar.write(f"GC Content range: {df['GC_Content'].min():.2f}% - {df['GC_Content'].max():.2f}%")

# -----------------------------
# Quick Stats
# -----------------------------
if not df.empty:
    st.subheader("📊 Sequence Statistics")
    st.markdown(f"- Total sequences: **{len(df)}**")
    st.markdown(f"- Average length: **{df['Length'].mean():.2f}**")
    st.markdown(f"- Average GC content: **{df['GC_Content'].mean():.2f}%**")
    
    st.subheader("Top 20 Longest Sequences")
    st.bar_chart(df.sort_values("Length", ascending=False).head(20)[["Length"]])

# -----------------------------
# Embeddings
# -----------------------------
@st.cache_resource
def embed_sequences(sequences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sequences, show_progress_bar=True)

embeddings = embed_sequences(df['Sequence'].tolist()) if not df.empty else None

# -----------------------------
# AI Chatbot
# -----------------------------
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def ai_chat(question, embeddings, sequences, top_k=5):
    if embeddings is None or len(sequences) == 0:
        return "No sequences available to answer."
    
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
    try:
        response = openai.chat.completions.create(
         model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": prompt}],
           temperature=0
        )

        answer = response.choices[0].message.content
        st.session_state['chat_history'].append({"Q": question, "A": answer})
        return answer
    except Exception as e:
        return f"OpenAI API Error: {e}"

# -----------------------------
# AI Chatbot UI
# -----------------------------
st.subheader("💬 AI Chatbot for eDNA Sequences")
user_question = st.text_input("Ask questions about your sequences")
if uploaded_files and user_question:
    answer = ai_chat(user_question, embeddings, df['Sequence'].tolist())
    st.markdown(f"**AI Answer:** {answer}")

# -----------------------------
# Downloadable Results
# -----------------------------
import io

# -----------------------------
# Downloadable Results
# -----------------------------
if not df.empty:
    # CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset CSV",
        data=csv_data,
        file_name='edna_analysis.csv',
        mime='text/csv'
    )

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='eDNA_Analysis')
    excel_buffer.seek(0)  # Move pointer to the start

    st.download_button(
        label="Download Dataset Excel",
        data=excel_buffer,
        file_name='edna_analysis.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
