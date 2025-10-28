import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# This is the corrected line 6
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
pplx_api_key = os.getenv("PERPLEXITY_API_KEY")

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Database (Vector Store) ---
PERSIST_DIR = "./chroma_db"
print("Loading knowledge base... this will take a moment.")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # Get top 5 results
print("Knowledge base loaded!")


# --- Initialize the LLM (Perplexity) ---
if not pplx_api_key:
    print("Error: PERPLEXITY_API_KEY not found in .env file.")
    # You might want to exit or raise an error here

# Use an offline Perplexity model to ensure only prompt context is used
llm = ChatPerplexity(model="llama-3-8b-instruct", pplx_api_key=pplx_api_key)

# --- RAG Prompt Template ---
template = """
You are a helpful medical AI assistant.
Answer the user's question based *only* on the context provided.
If the answer is not in the context, say "I could not find that information in the documents."

CONTEXT:
{context}

QUESTION:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Create the API Endpoint ---
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # --- Run the RAG Chain ---
        # 1. Find relevant documents
        relevant_docs = retriever.invoke(user_prompt)

        # 2. Format the prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        formatted_prompt = prompt.format(context=context_text, question=user_prompt)

        # 3. Get the answer
        response = llm.invoke(formatted_prompt)
        answer = response.content

        # 4. Format sources
        sources = []
        for doc in relevant_docs:
            sources.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "content": doc.page_content[:300] + "..."
            })

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        # Handle API key errors or other issues
        return jsonify({"error": str(e)}), 500

# This is just a test route
@app.route("/")
def serve_index():
    # Serve index.html from the same directory as this file
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

if __name__ == "__main__":
    app.run(port=5000, debug=True)