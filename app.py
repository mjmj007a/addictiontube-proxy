from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
import os
import re
import tiktoken
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("addictiontube-index")

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '')

@app.route('/search_stories', methods=['GET'])
def search_stories():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    page = int(request.args.get('page', 1))
    size = int(request.args.get('per_page', 5))

    if not query or not category:
        return jsonify({"error": "Missing query or category"}), 400

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        return jsonify({"error": "OpenAI embedding failed", "details": str(e)}), 500

    try:
        results = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )
        total = len(results.matches)
        start = (page - 1) * size
        end = start + size
        paginated = results.matches[start:end]
        stories = [{
            "id": m.id,
            "score": m.score,
            "title": m.metadata.get("title", "N/A"),
            "description": m.metadata.get("description", ""),
            "image": m.metadata.get("image", "")
        } for m in paginated]
        return jsonify({"results": stories, "total": total})
    except Exception as e:
        return jsonify({"error": "Pinecone query failed", "details": str(e)}), 500

@app.route('/rag_answer', methods=['GET'])
def rag_answer():
    query = request.args.get('q', '')
    category = request.args.get('category', '')

    if not query or not category:
        return jsonify({"error": "Missing query or category"}), 400

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        return jsonify({"error": "Embedding failed", "details": str(e)}), 500

    try:
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )

        context_docs = [
            strip_html(match.metadata.get("text", ""))[:3000]
            for match in results.matches
        ]

        encoding = tiktoken.encoding_for_model("gpt-4")
        total_tokens = sum(len(encoding.encode(doc)) for doc in context_docs)
        print("DEBUG: total_tokens =", total_tokens)

        context_text = "\n\n---\n\n".join(context_docs)

        system_prompt = "You are an expert addiction recovery assistant."
        user_prompt = f"""Use the following recovery stories to answer the question.

{context_text}

Question: {query}
Answer:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content.replace("—", ", ")
        return jsonify({"answer": answer})
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print("ERROR during OpenAI ChatCompletion:", traceback_str)
        return jsonify({
            "error": "RAG processing failed",
            "details": str(e),
            "traceback": traceback_str
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
