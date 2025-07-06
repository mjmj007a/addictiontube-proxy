from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
import os
import re
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# Validate environment variables
required_env = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
for var in required_env:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing environment variable: {var}")

# Configure logging
logger = logging.getLogger('addictiontube')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('story_image_debug.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(handler)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

try:
    client = OpenAI(timeout=30)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("addictiontube-index")

try:
    import tiktoken
    tiktoken_available = True
    logger.info("tiktoken imported successfully")
except ImportError:
    tiktoken_available = False
    logger.error("tiktoken import failed")

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '')

@app.route('/search_stories', methods=['GET'])
def search_stories():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    category = request.args.get('category', '')
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    if not query or not category or category not in ['1028', '1042']:
        return jsonify({"error": "Invalid or missing query or category"}), 400

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {str(e)}")
        return jsonify({"error": "Embedding service unavailable"}), 500

    try:
        top_k = min(100, size * page)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )
        total = len(results.matches)
        start = (page - 1) * size
        end = start + size
        paginated = results.matches[start:end]

        stories = []
        for m in paginated:
            stories.append({
                "id": m.id,
                "score": m.score,
                "title": strip_html(m.metadata.get("title", "N/A")),
                "description": strip_html(m.metadata.get("description", "")),
                "image": m.metadata.get("image", "")
            })
        return jsonify({"results": stories, "total": total})
    except Exception as e:
        logger.error(f"Pinecone query failed: {str(e)}")
        return jsonify({"error": "Search service unavailable"}), 500

@app.route('/rag_answer', methods=['GET'])
def rag_answer():
    if not tiktoken_available:
        return jsonify({"error": "AI answer service unavailable due to missing tiktoken"}), 503

    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    category = request.args.get('category', '')

    if not query or not category or category not in ['1028', '1042']:
        return jsonify({"error": "Invalid or missing query or category"}), 400

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return jsonify({"error": "Embedding service unavailable"}), 500

    try:
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )

        if not results.matches:
            return jsonify({"error": "No relevant context found"}), 404

        encoding = tiktoken.get_encoding("cl100k_base")  # Compatible with tiktoken 0.9.0
        max_tokens = 16384 - 1000
        context_docs = []
        total_tokens = 0
        for match in results.matches:
            text = match.metadata.get("text", "")
            if not text:
                logger.warning(f"Match {match.id} has no text metadata")
                continue
            doc = strip_html(text)[:3000]
            doc_tokens = len(encoding.encode(doc))
            if total_tokens + doc_tokens <= max_tokens:
                context_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        if not context_docs:
            return jsonify({"error": "No usable context data found"}), 404

        logger.debug(f"Total tokens: {total_tokens}")
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
            ],
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except openai.OpenAIError as e:
        logger.error(f"OpenAI error in rag_answer: {str(e)}")
        return jsonify({"error": f"OpenAI error: {str(e)}"}), 500
    except pinecone.PineconeException as e:
        logger.error(f"Pinecone error in rag_answer: {str(e)}")
        return jsonify({"error": f"Pinecone error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"RAG processing failed: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)