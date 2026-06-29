import math
import nltk
from flask import Flask, request, jsonify

nltk.download("punkt_tab", quiet=True)

app = Flask(__name__)


def score_burstiness(text: str) -> float:
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 3:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.5

    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean

    # High CV (irregular lengths) → human-like → low score.
    # Invert so that uniform AI text (CV ≈ 0) maps to score ≈ 1.0.
    score = 1.0 - min(cv / 2.0, 1.0)
    return round(max(0.0, min(1.0, score)), 4)


@app.route("/")
def home():
    return "Provenance Guard is running."


@app.route("/submit", methods=["POST"])
def submit():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    content_id = body.get("kcontent_id")
    text = body.get("text")

    if not content_id or not text:
        return jsonify({"error": "Both 'content_id' and 'text' are required"}), 400

    signal_1_score = score_burstiness(text)
    return jsonify({"content_id": content_id, "signal_1_score": signal_1_score})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
