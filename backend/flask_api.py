from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from dotenv import load_dotenv
from model.check_rephrase_content import *
from model.check_generate_content import *
from flask import request, jsonify
from werkzeug.utils import secure_filename
# ✅ 初始化：讀取環境變數與定義全域常數
load_dotenv()

REQUIRED_ENV_VARS = [
    "OPENAI_APIKEY",
    "BACKEND_API_URL_TEXT_GENERATE",
    "BACKEND_API_URL_TEXT_REPHRASE",
    "BACKEND_API_URL_PDF_GENERATE",
    "BACKEND_API_URL_PDF_REPHRASE",
]

SOURCE_DIRS = [
    # 正式部署用
    # "dataset/paraphrased_dataset/source/ncu_2019",
    # "dataset/paraphrased_dataset/source/ncu_2020",

    # 測試用路徑
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
]


# ✅ 檢查環境變數是否完整
def validate_env():
    # 檢查環境變數
    missing = [var for var in REQUIRED_ENV_VARS if os.getenv(var) is None]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}")

    # 檢查資料夾是否存在
    missing_dirs = [d for d in SOURCE_DIRS if not os.path.exists(d)]
    if missing_dirs:
        raise FileNotFoundError(
            f"Missing required source directories: {', '.join(missing_dirs)}")

    print("✅ Environment variables loaded.")
    print("✅ All source directories exist.")


# ✅ 初始化應用程式（Flask app 與資料）
def create_app():
    app = Flask(__name__)

    CORS(app)  # 跨來源請求支援

    validate_env()

    print(
        "Building FAISS vector store with all files... (this may take a while)"
    )
    global_vector_db, global_embedding_model, _ = build_vector_db(SOURCE_DIRS)

    # --- 路由定義區 ---
    @app.route('/')
    def home():
        return "Plagiarism Checker API is running!"

    @app.route(os.getenv("BACKEND_API_URL_TEXT_REPHRASE"), methods=['POST'])
    def upload_text_rephrase():
        return text_rephrase_check(request, global_vector_db,
                                   global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_REPHRASE"), methods=['POST'])
    def upload_pdf_rephrase():
        return pdf_rephrase_check(request, global_vector_db,
                                  global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_TEXT_GENERATE"), methods=['POST'])
    def upload_text_generate():
        return simulate_plagiarism_check(request, global_vector_db,
                                         global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_GENERATE"), methods=['POST'])
    def upload_pdf_generate():
        return pdf_rephrase_check(request, global_vector_db,
                                  global_embedding_model)

    return app


def text_rephrase_check(req, global_vector_db, global_embedding_model):
    data = req.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Check if global vector database is available
    if global_vector_db is None:
        return jsonify({"error": "Vector database not initialized"}), 500

    # Run plagiarism check
    result = cooperate_plagiarism_check(user_text=text,
                                        vector_db=global_vector_db,
                                        embedding_model=global_embedding_model)

    return result


def pdf_rephrase_check(req, global_vector_db, global_embedding_model):
    if 'file' not in req.files:
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_file = req.files['file']
    original_filename = uploaded_file.filename

    if not original_filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # 避免危險字元，轉成安全檔名
        safe_filename = secure_filename(original_filename)

        save_dir = "./backend/uploaded_pdfs"
        os.makedirs(save_dir, exist_ok=True)

        saved_path = os.path.join(save_dir, safe_filename)
        uploaded_file.save(saved_path)

    except Exception as e:
        return jsonify({"error": f"Failed to save PDF: {str(e)}"}), 500

    # ///////////////////////////////////////////////////////////////////
    text = "這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應"
    plagiarism_percentage = round(random.uniform(0, 100), 2)
    confidence_score = round(random.uniform(0, 100), 2)
    plagiarism_snippet = text[:min(30, len(text))]

    return jsonify({
        "plagiarism_percentage": plagiarism_percentage,
        "plagiarism_snippet": plagiarism_snippet,
        "confidence_score": confidence_score,
    })


# ✅ 模擬抄襲檢測回應（之後可以替換為真實邏輯）
def simulate_plagiarism_check(req, global_vector_db, global_embedding_model):
    data = req.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    plagiarism_percentage = round(random.uniform(0, 100), 2)
    confidence_score = round(random.uniform(0, 100), 2)
    plagiarism_snippet = text[:min(30, len(text))]

    return jsonify({
        "plagiarism_percentage": plagiarism_percentage,
        "plagiarism_snippet": plagiarism_snippet,
        "confidence_score": confidence_score,
    })


app = create_app()
# ✅ 若是直接執行此檔案，則啟動開發伺服器
if __name__ == "__main__":
    load_dotenv()
    app.run(host="0.0.0.0", port=8077, debug=True)
