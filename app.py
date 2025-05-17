import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
from tqdm import tqdm

# Initialize OpenAI client (via OpenRouter)
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="📘 AI Textbook Assistant", layout="wide")
st.title("📘 AI Textbook Question Answering App")
st.markdown("Upload a textbook (PDF), ask any question, and get a smart AI answer!")

# -------------------------------
# PDF Upload & Parsing
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload your textbook (PDF)", type="pdf")

if uploaded_file:
    # Read PDF text
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    st.success("✅ PDF loaded successfully!")

    # Split into paragraphs for embeddings
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    embeddings = model.encode(paragraphs, convert_to_tensor=True)

    st.info(f"🔍 Indexed {len(paragraphs)} paragraphs for answering.")

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
            model="openai/gpt-3.5-turbo",  # Or gpt-4 if you have access
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    if st.button("🧠 Get Answer") and question:
        # Embed the question
        question_embedding = model.encode(question, convert_to_tensor=True)
        # Find top relevant paragraphs
        hits = util.semantic_search(question_embedding, embeddings, top_k=5)
        top_indices = hits[0]
        context = "\n\n".join([paragraphs[idx['corpus_id']] for idx in top_indices])
        answer = ask_gpt(context, question)
        st.success("✅ Answer:")
        st.write(answer)

    # -------------------------------
    # Summarization Feature
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
