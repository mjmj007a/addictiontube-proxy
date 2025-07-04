from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from pinecone import Pinecone
import os
import re
import tiktoken
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("addictiontube-index")

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '')

@app.route('/api/search', methods=['GET'])
def search_stories():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    page = int(request.args.get('page', 1))
    size = int(request.args.get('per_page', 5))

    if not query or not category:
        return jsonify({"error": "Missing query or category"}), 400

    try:
        embedding = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']

        results = index.query(
            vector=embedding,
            top_k=100,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )

        matches = results.matches
        paginated = matches[(page-1)*size : page*size]

        stories = []
        for m in paginated:
            img = m.metadata.get("image", "")
            if img and not img.startswith("http"):
                img = "https://addictiontube.com/" + img.lstrip("/")
            stories.append({
                "id": m.id,
                "score": m.score,
                "title": m.metadata.get("title", ""),
                "description": m.metadata.get("description", ""),
                "image": img
            })

        return jsonify({"results": stories, "total": len(matches)})
    except Exception as e:
        return jsonify({"error": "Search failed", "details": str(e)}), 500

@app.route('/api/answer', methods=['GET'])
def rag_answer():
    query = request.args.get('q', '')
    category = request.args.get('category', '')

    if not query or not category:
        return jsonify({"error": "Missing query or category"}), 400

    try:
        embedding = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )['data'][0]['embedding']

        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )

        context_docs = [
            strip_html(m.metadata.get("text", ""))[:3000]
            for m in results.matches
        ]

        encoding = tiktoken.encoding_for_model("gpt-4")
        total_tokens = sum(len(encoding.encode(doc)) for doc in context_docs)
        print(f"[DEBUG] Context tokens: {total_tokens}")

        context_block = "\n\n---\n\n".join(context_docs)
        user_prompt = f"""Use the following recovery stories to answer the question:

{context_block}

Question: {query}
Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert addiction recovery assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response['choices'][0]['message']['content'].replace("—", ", ")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": "RAG failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
