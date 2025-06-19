import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import numpy as np
from tqdm import tqdm

# ---------------------------
# Configure OpenAI Client via OpenRouter
# ---------------------------
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="📘 AI Textbook Assistant", layout="wide")
st.title("📘 AI Textbook Question Answering App")
st.markdown("Upload a textbook (PDF), ask any question, and get a smart AI answer!")

# ---------------------------
# PDF Upload & Parsing
# ---------------------------
uploaded_file = st.file_uploader("📄 Upload your textbook (PDF)", type="pdf")

if uploaded_file:
    # Extract text from PDF
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF loaded and parsed successfully!")

    # Break text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

    # ---------------------------
    # Embedding with OpenAI
    # ---------------------------
    def get_embedding(text):
        response = client.embeddings.create(
            model="openai/text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    with st.spinner("🔍 Generating paragraph embeddings..."):
        para_embeddings = [get_embedding(p) for p in paragraphs]
        st.success(f"✅ Indexed {len(para_embeddings)} paragraphs.")

    # ---------------------------
    # Ask a Question
    # ---------------------------
    question = st.text_input("❓ Ask a question from the textbook")

    def ask_gpt(context, question):
        prompt = f"""You are a helpful AI tutor. Answer the question based on the context below.

        Context:
        {context}

        Question:
        {question}

        Answer:"""

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    if st.button("🧠 Get Answer") and question:
        question_embedding = get_embedding(question)
        scores = [cosine_similarity(question_embedding, emb) for e]()
