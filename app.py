from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
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
        embedding_response = openai.embeddings.create(
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
            "description": m.metadata.get("description", "")
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
        embedding_response = openai.embeddings.create(
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
            strip_html(match.metadata.get("text", ""))[:10000]  # Limit each doc to ~10,000 characters (~2,000 words)
            for match in results.matches
        ]
        print("DEBUG: context_docs", context_docs)

        context_text = "\n\n---\n\n".join(context_docs)

        system_prompt = "You are an expert addiction recovery assistant."
        user_prompt = f"""Use the following recovery stories to answer the question.

{context_text}

Question: {query}
Answer:"""

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = completion['choices'][0]['message']['content']
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": "RAG processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
