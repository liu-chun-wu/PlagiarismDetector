from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)  # 允許跨來源請求，讓 React 能夠訪問 API


@app.route('/')
def home():
    return "Flask API is running!"


@app.route('/upload', methods=['POST'])
def upload_text():
    data = request.json  # 確保前端傳來的是 JSON
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 模擬抄襲檢測（實際應用可以使用 NLP 或 AI 模型來處理）
    plagiarism_percentage = round(random.uniform(0, 100), 2)
    confidence_score = round(random.uniform(0, 100), 2)

    # 模擬回傳抄襲的片段（這裡假裝隨機回傳一部分文字）
    plagiarism_snippet = text[:min(30, len(text))]

    response = {
        "plagiarism_percentage": plagiarism_percentage,
        "plagiarism_snippet": plagiarism_snippet,
        "confidence_score": confidence_score,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
