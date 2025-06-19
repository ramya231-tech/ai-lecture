import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import os
from tqdm import tqdm
import numpy as np

# Set up OpenAI client using OpenRouter
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Configure Streamlit UI
st.set_page_config(page_title="📘 AI Textbook Assistant", layout="wide")
st.title("📘 AI Textbook Question Answering App")
st.markdown("Upload a textbook (PDF), ask any question, and get a smart AI answer!")

# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload your textbook (PDF)", type="pdf")

if uploaded_file:
    # Read and extract text
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF loaded successfully!")

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

    # -------------------------------
    # Generate embeddings using OpenAI
    # -------------------------------
    def get_embedding(text):
        response = client.embeddings.create(
            model="openai/text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    with st.spinner("🔍 Generating embeddings..."):
        para_embeddings = [get_embedding(p) for p in paragraphs]
        st.success(f"✅ {len(para_embeddings)} paragraphs indexed.")

    # -------------------------------
    # Question Answering
    # -------------------------------
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
        q_embedding = get_embedding(question)
        scores = [cosine_similarity(q_embedding, emb) for emb in para_embeddings]
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        context = "\n\n".join([paragraphs[i] for i, _ in top_k])
        answer = ask_gpt(context, question)
        st.success("✅ Answer:")
        st.write(answer)

    # -------------------------------
    # Summarization
    # -------------------------------
    if st.button("📌 Summarize Entire Book"):
        st.info("⏳ Summarizing book...")
        all_chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
        full_summary = ""

        for chunk in tqdm(all_chunks, desc="Summarizing", leave=False):
            prompt = f"Summarize this text:\n\n{chunk}"
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            full_summary += response.choices[0].message.content.strip() + "\n\n"

        st.success("✅ Summary Ready:")
        st.write(full_summary)
