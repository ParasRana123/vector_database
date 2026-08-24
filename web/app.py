from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = PROJECT_ROOT / "build" / "Release" / "test_onnx.exe"
EMBEDDING_EXECUTABLE = Path(os.environ.get("VECTOR_DB_EXECUTABLE", DEFAULT_EXECUTABLE))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-vector-db-demo")


@dataclass
class Document:
    id: int
    text: str
    embedding: list[float]


documents: list[Document] = []
next_document_id = 1


def embed(text: str) -> list[float]:
    """Generate one MiniLM embedding through the C++ ONNX executable."""
    if not EMBEDDING_EXECUTABLE.is_file():
        raise RuntimeError(
            "MiniLM executable was not found. Build it first with: "
            "cmake --build build --config Release --target test_onnx"
        )

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
        raise ValueError("Vectors must be non-empty and have matching dimensions.")
    dot = sum(left * right for left, right in zip(a, b))
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    return dot / (a_norm * b_norm)


@app.get("/")
def index():
    return render_template("index.html", documents=documents, results=None, query="machine learning")


@app.post("/documents")
def add_document():
    global next_document_id
    text = request.form.get("text", "").strip()
    if not text:
        flash("Enter some text before adding a document.", "error")
    else:
        try:
            documents.append(Document(next_document_id, text, embed(text)))
            next_document_id += 1
            flash("Document added and embedded with MiniLM.", "success")
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            flash(str(error), "error")
    return redirect(url_for("index"))


@app.post("/search")
def search():
    query = request.form.get("query", "").strip()
    if not query or not documents:
        flash("Add at least one document and enter a query.", "error")
        return render_template("index.html", documents=documents, results=None, query=query)
    try:
        query_embedding = embed(query)
        results = sorted(
            (
                {"id": document.id, "text": document.text,
                 "score": cosine_similarity(query_embedding, document.embedding)}
                for document in documents
            ),
            key=lambda result: result["score"],
            reverse=True,
        )
        return render_template("index.html", documents=documents, results=results, query=query)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        flash(str(error), "error")
        return render_template("index.html", documents=documents, results=None, query=query)


@app.post("/example")
def load_example():
    global next_document_id
    example_texts = [
        "I study artificial intelligence",
        "I like playing football",
        "Deep learning is a branch of machine learning",
    ]
    try:
        for text in example_texts:
            documents.append(Document(next_document_id, text, embed(text)))
            next_document_id += 1
        flash("Example documents added. Search for 'machine learning'.", "success")
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        flash(str(error), "error")
    return redirect(url_for("index"))


@app.post("/clear")
def clear_documents():
    global next_document_id
    documents.clear()
    next_document_id = 1
    flash("All in-memory documents were cleared.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
