from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
import requests

from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "build" / "Release" / "test_onnx.exe"
EMBEDDING_EXECUTABLE = Path(os.environ.get("VECTOR_DB_EXECUTABLE", DEFAULT_EXECUTABLE))
VECTOR_DB_API_URL = (
    os.environ.get("VECTOR_DB_API_URL")
    or os.environ.get("VECTORDB_BACKEND_URL")
    or ""
).rstrip("/")

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"))
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-vector-db-demo")


@app.route("/static/<path:filename>")
def custom_static(filename: str):
    static_dir = Path(__file__).resolve().parent / "static"
    return send_from_directory(str(static_dir), filename)


@dataclass
class Document:
    id: int
    text: str
    embedding: list[float]


documents: list[Document] = []
next_document_id = 1


def is_remote_backend() -> bool:
    return bool(VECTOR_DB_API_URL)


def embed_local(text: str) -> list[float]:
    """Generate one MiniLM embedding through the local C++ ONNX executable."""
    if not EMBEDDING_EXECUTABLE.is_file():
        # Fallback hash embedding if binary not present (e.g. on serverless Vercel)
        import hashlib, random
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(384)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    completed = subprocess.run(
        [str(EMBEDDING_EXECUTABLE), "--embed", text],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "MiniLM embedding process failed.")

    try:
        values = [float(value) for value in completed.stdout.strip().split(",")]
    except ValueError as error:
        raise RuntimeError("MiniLM returned an invalid embedding.") from error
    if len(values) != 384:
        raise RuntimeError(f"Expected a 384-dimensional MiniLM embedding, received {len(values)} values.")
    return values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


@app.get("/")
def index():
    return render_template(
        "index.html",
        documents=documents,
        results=None,
        query="machine learning",
        is_remote=is_remote_backend()
    )


@app.post("/documents")
def add_document():
    global next_document_id
    text = request.form.get("text", "").strip()
    if not text:
        flash("Enter some text before adding a document.", "error")
        return redirect(url_for("index"))

    if is_remote_backend():
        try:
            resp = requests.post(
                f"{VECTOR_DB_API_URL}/api/insert",
                json={"collection": "default", "id": next_document_id, "text": text},
                timeout=10,
            )
            if resp.status_code == 200:
                documents.append(Document(next_document_id, text, []))
                next_document_id += 1
                flash("Document indexed in remote C++ VectorDB (Render).", "success")
            else:
                flash(f"Remote DB error: {resp.text}", "error")
        except Exception as error:
            flash(f"Failed to reach remote VectorDB: {error}", "error")
    else:
        try:
            documents.append(Document(next_document_id, text, embed_local(text)))
            next_document_id += 1
            flash("Document added and embedded with MiniLM.", "success")
        except Exception as error:
            flash(str(error), "error")

    return redirect(url_for("index"))


@app.post("/search")
def search():
    query = request.form.get("query", "").strip()
    if not query:
        flash("Enter a query to search.", "error")
        return render_template(
            "index.html",
            documents=documents,
            results=None,
            query=query,
            is_remote=is_remote_backend()
        )

    if is_remote_backend():
        try:
            resp = requests.post(
                f"{VECTOR_DB_API_URL}/api/search",
                json={"collection": "default", "query": query, "top_k": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = [
                    {"id": r["id"], "text": r["payload"], "score": round(r["score"], 4)}
                    for r in data.get("results", [])
                ]
                latency = data.get("latency_ms", 0.0)
                flash(f"Search completed in {latency:.2f} ms via C++ VectorDB on Render.", "info")
                return render_template(
                    "index.html",
                    documents=documents,
                    results=results,
                    query=query,
                    is_remote=is_remote_backend()
                )
            else:
                flash(f"Remote search error: {resp.text}", "error")
        except Exception as error:
            flash(f"Failed to connect to remote VectorDB: {error}", "error")
        return render_template(
            "index.html",
            documents=documents,
            results=None,
            query=query,
            is_remote=is_remote_backend()
        )
    else:
        if not documents:
            flash("Add at least one document before searching.", "error")
            return render_template(
                "index.html",
                documents=documents,
                results=None,
                query=query,
                is_remote=is_remote_backend()
            )
        try:
            query_embedding = embed_local(query)
            results = sorted(
                (
                    {"id": document.id, "text": document.text,
                     "score": round(cosine_similarity(query_embedding, document.embedding), 4)}
                    for document in documents
                ),
                key=lambda result: result["score"],
                reverse=True,
            )
            return render_template(
                "index.html",
                documents=documents,
                results=results,
                query=query,
                is_remote=is_remote_backend()
            )
        except Exception as error:
            flash(str(error), "error")
            return render_template(
                "index.html",
                documents=documents,
                results=None,
                query=query,
                is_remote=is_remote_backend()
            )


@app.post("/example")
def load_example():
    global next_document_id
    example_texts = [
        "I study artificial intelligence and neural networks",
        "I like playing football and outdoor sports",
        "Deep learning is a branch of machine learning and data science",
    ]
    for text in example_texts:
        if is_remote_backend():
            try:
                requests.post(
                    f"{VECTOR_DB_API_URL}/api/insert",
                    json={"collection": "default", "id": next_document_id, "text": text},
                    timeout=5,
                )
            except Exception:
                pass
        documents.append(Document(next_document_id, text, [] if is_remote_backend() else embed_local(text)))
        next_document_id += 1

    flash("Example documents added. Search for 'machine learning'.", "success")
    return redirect(url_for("index"))


@app.post("/clear")
def clear_documents():
    global next_document_id
    documents.clear()
    next_document_id = 1
    flash("All documents were cleared.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
