import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pypdf import PdfReader
import numpy as np
import os

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are an intelligent and helpful AI assistant designed for the Department Chatbot of CSJMU (Chhatrapati Shahu Ji Maharaj University), Kanpur.

Your role is to answer student queries related to:
- B.Tech admissions
- Academic calendar
- Courses and programs
- Eligibility criteria
- Campus facilities
- Exams, results, and academic activities

STRICT INSTRUCTIONS:

1. SOURCE-BASED ANSWERS:
- Always answer ONLY from the provided PDF documents (context).
- Do NOT generate answers from your own knowledge.
- If answer is found, respond clearly and accurately.

2. IF ANSWER NOT FOUND:
- If the answer is not available in the given context, respond with:
  "I could not find the answer in the provided documents. Please update the knowledge base."
- Do NOT guess or hallucinate.

3. LANGUAGE STYLE:
- Keep answers simple, clear, and student-friendly.
- Use Hinglish if needed (mix of Hindi + English) for better understanding.
- Avoid very long paragraphs.

4. CONTEXT HANDLING:
- Use only relevant information from retrieved chunks.
- If multiple pieces of info are found, combine them properly.

5. ACCURACY RULE:
- Dates, eligibility, and numbers must be EXACT.
- Example: Academic calendar dates (like exams, registration) must match exactly.

6. DOMAIN LIMIT:
- Only answer questions related to CSJMU, UIET, admissions, academics.
- If question is outside domain, respond:
  "This question is outside my scope. I can help with CSJMU-related queries only."

7. FORMATTING:
- Use bullet points where needed.
- Keep answers structured.

8. NO EXTRA INFORMATION:
- Do not add assumptions or external knowledge.
- Stick strictly to provided documents.

Your goal is to act like a reliable official university assistant.
"""
# ---------------- UI ----------------
st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("🤖 CSE Smart Assistant (PDF + Manual + Default Data)")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CLEAR HISTORY ----------------
if st.button("🗑️ Clear Chat History"):
    st.session_state.history = []

# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader("📄 Upload PDF (optional)", type="pdf")

# ---------------- LOAD DEFAULT DATA ----------------
def load_default_data():
    text = ""
    if os.path.exists("data"):
        for file in os.listdir("data"):
            file_path = os.path.join("data", file)

            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"

            elif file.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    return text

# ---------------- MANUAL DATA ----------------
def load_manual_data():
    return """
SLM department timing is 9 AM to 5 PM.
HOD of SLM department is Dr. XYZ.
CSE department offers subjects like DBMS, OS, CN, AI.
Lab facilities include programming lab, networking lab.
"""

# ---------------- PDF TEXT ----------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---------------- CHUNKING ----------------
def split_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- SYSTEM SETUP ----------------
@st.cache_resource
def setup_system(text_chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(text_chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        torch_dtype="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.2
    )

    return text_chunks, embedder, index, generator

# ---------------- DATA SELECTION ----------------
manual_text = load_manual_data()
default_text = load_default_data()

if uploaded_file:
    st.info("📄 Using uploaded PDF + manual data")
    pdf_text = extract_text_from_pdf(uploaded_file)
    combined_text = pdf_text + "\n" + manual_text
else:
    st.info("📚 Using default data + manual data")
    combined_text = default_text + "\n" + manual_text

# ---------------- CHUNK ----------------
text_chunks = split_text(combined_text)

# ---------------- SETUP ----------------
if text_chunks:
    docs, embedder, index, generator = setup_system(text_chunks)

    query = st.text_input("💬 Ask your question:")

    if query:
        q_emb = embedder.encode([query])
        distances, idx = index.search(np.array(q_emb), 3)

        relevant_chunks = [docs[i] for i in idx[0]]
        context = "\n".join(relevant_chunks)

        # ❌ If no relevant answer
        if distances[0][0] > 1.5:
            response = "❌ Answer not found in the available data."
            st.warning(response)

        else:
            prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question: {query}

Answer:
"""

            result = generator(prompt)[0]["generated_text"]

            # Clean output
            response = result.replace(prompt, "").strip()

            # Safety check
            if response == "" or len(response) < 3:
                response = "Answer not found"

            st.success(response)

        # Save history
        st.session_state.history.append((query, response))

# ---------------- CHAT HISTORY ----------------
st.subheader("🕘 Chat History")

if len(st.session_state.history) == 0:
    st.info("No chat history yet.")
else:
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.markdown("---")