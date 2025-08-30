import streamlit as st
from Bio import SeqIO, Phylo
from Bio.SeqUtils import molecular_weight, MeltingTemp as mt
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align.Applications import ClustalwCommandline
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io
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
# Parse FASTA with extended features
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
            "GC_Content": [100 * (rec.seq.count("G") + rec.seq.count("C")) / len(rec.seq) for rec in records],
            "AT_Content": [100 * (rec.seq.count("A") + rec.seq.count("T")) / len(rec.seq) for rec in records],
            "N_Content": [100 * rec.seq.count("N") / len(rec.seq) for rec in records],
            "Molecular_Weight": [molecular_weight(rec.seq, "DNA") for rec in records],
            "Tm_Wallace": [mt.Tm_Wallace(rec.seq) for rec in records],
            "Reverse_Complement": [str(rec.seq.reverse_complement()) for rec in records]
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
# Tabs for Stats, Sequences, AI, BLAST, Tree
# -----------------------------
if not df.empty:
    tabs = st.tabs(["📊 Stats", "🧬 Sequences", "💬 AI Chatbot", "🔬 BLAST", "🌳 Phylogenetic Tree"])

    # -----------------------------
    # Stats Tab
    # -----------------------------
    with tabs[0]:
        st.subheader("Sequence Statistics")
        st.markdown(f"- Total sequences: **{len(df)}**")
        st.markdown(f"- Average length: **{df['Length'].mean():.2f}**")
        st.markdown(f"- Average GC content: **{df['GC_Content'].mean():.2f}%**")
        st.markdown(f"- Average AT content: **{df['AT_Content'].mean():.2f}%**")
        st.markdown(f"- Average N content: **{df['N_Content'].mean():.2f}%**")
        st.markdown(f"- Average Molecular Weight: **{df['Molecular_Weight'].mean():.2f} Da**")
        st.markdown(f"- Average Melting Temp (Tm): **{df['Tm_Wallace'].mean():.2f} °C**")
        st.subheader("Top 20 Longest Sequences")
        st.bar_chart(df.sort_values("Length", ascending=False).head(20)[["Length"]])

    # -----------------------------
    # Sequences Tab
    # -----------------------------
    with tabs[1]:
        st.subheader("Sequences Preview")
        st.dataframe(df)

    # -----------------------------
    # AI Chatbot Tab
    # -----------------------------
    with tabs[2]:
        st.subheader("AI Chatbot for eDNA Sequences")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        @st.cache_resource
        def embed_sequences(sequences):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(sequences, show_progress_bar=True)

        embeddings = embed_sequences(df['Sequence'].tolist())

        user_question = st.text_input("Ask questions about your sequences")
        if user_question:
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

            answer = ai_chat(user_question, embeddings, df['Sequence'].tolist())
            st.markdown(f"**AI Answer:** {answer}")

    # -----------------------------
    # BLAST Tab
    # -----------------------------
    with tabs[3]:
        st.subheader("BLAST Search")
        seq_to_blast = st.selectbox("Select sequence for BLAST", df['Sequence'])
        if st.button("Run BLAST"):
            try:
                st.info("Running BLAST search. This may take some time...")
                result_handle = NCBIWWW.qblast("blastn", "nt", seq_to_blast)
                blast_record = NCBIXML.read(result_handle)
                top_hits = []
                for alignment in blast_record.alignments[:5]:
                    for hsp in alignment.hsps:
                        top_hits.append({
                            "Title": alignment.title,
                            "Length": alignment.length,
                            "Score": hsp.score,
                            "E-value": hsp.expect
                        })
                st.dataframe(pd.DataFrame(top_hits))
            except Exception as e:
                st.error(f"BLAST Error: {e}")

    # -----------------------------
    # Phylogenetic Tree Tab
    # -----------------------------
    with tabs[4]:
        st.subheader("Phylogenetic Tree")
        if st.button("Build Phylogenetic Tree"):
            try:
                # Save sequences temporarily
                fasta_temp = "temp_sequences.fasta"
                with open(fasta_temp, "w") as f:
                    for i, row in df.iterrows():
                        f.write(f">{row['ID']}\n{row['Sequence']}\n")
                
                # Run ClustalW (must be installed locally)
                clustalw_cline = ClustalwCommandline("clustalw2", infile=fasta_temp)
                stdout, stderr = clustalw_cline()
                
                tree_file = fasta_temp.replace(".fasta", ".dnd")
                tree = Phylo.read(tree_file, "newick")
                fig = plt.figure(figsize=(8, 5))
                Phylo.draw(tree, do_show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Phylogenetic Tree Error: {e}")

# -----------------------------
# Downloadable Results
# -----------------------------
if not df.empty:
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset CSV",
        data=csv_data,
        file_name='edna_analysis.csv',
        mime='text/csv'
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='eDNA_Analysis')
    excel_buffer.seek(0)
    st.download_button(
        label="Download Dataset Excel",
        data=excel_buffer,
        file_name='edna_analysis.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
