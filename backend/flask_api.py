from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from dotenv import load_dotenv
from model.check_rephrase_content import *
from model.check_generate_content import *
from pdf_processor_SogoChang.pdf_cutter import *
from pdf_processor_SogoChang.json_to_array import *
from pdf_processor_SogoChang.array_to_json import *
import logging

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
    "dataset/paraphrased_dataset/source/ncu_2019",
    "dataset/paraphrased_dataset/source/ncu_2020",
    "dataset/paraphrased_dataset/source/ccu",
    "dataset/paraphrased_dataset/source/nycu",

    # 測試用路徑
    # "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    # "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
    # "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ccu",
    # "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nycu"
]

# 正式部署用
PDF_SAVE_DIR = "uploaded_pdfs"

# 測試用路徑
# PDF_SAVE_DIR = "/home/undergrad/PlagiarismDetector/backend/uploaded_pdfs"


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

    # 設定 logger 級別為 DEBUG
    app.logger.setLevel(logging.DEBUG)

    # handler = logging.StreamHandler()
    # handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    # handler.setFormatter(formatter)
    # app.logger.addHandler(handler)

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
        return rephrase_text_check(request, global_vector_db,
                                   global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_REPHRASE"), methods=['POST'])
    def upload_pdf_rephrase():
        return rephrase_pdf_check(request, global_vector_db,
                                  global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_TEXT_GENERATE"), methods=['POST'])
    def upload_text_generate():
        return simulate_plagiarism_check(request, global_vector_db,
                                         global_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_GENERATE"), methods=['POST'])
    def upload_pdf_generate():
        return simulate_generate_pdf(request, global_vector_db,
                                     global_embedding_model)

    return app


app = create_app()


def rephrase_text_check(req, global_vector_db, global_embedding_model):
    app.logger.debug("📥 收到文字 Rephrase 請求")

    try:
        data = req.json
        text = data.get("text", "")
        app.logger.debug(f"📄 傳入文字：{text[:50]}...")

        if not text:
            app.logger.warning("⚠️ 沒有提供文字")
            return jsonify({"error": "No text provided"}), 400

        if global_vector_db is None:
            app.logger.error("❌ Vector database 未初始化")
            return jsonify({"error": "Vector database not initialized"}), 500

        check_paragraph_result = cooperate_plagiarism_check(
            user_text=text,
            vector_db=global_vector_db,
            embedding_model=global_embedding_model)

        app.logger.debug("✅ 檢測完成，開始組裝結果")

        result = {
            "plagiarism_percentage":
            round(check_paragraph_result["plagiarism_percentage"], 2),
            "plagiarism_confidence":
            round(check_paragraph_result["plagiarism_confidence"], 2),
            "original_text_and_plagiarism_snippet": [{
                "original_text":
                text,
                "plagiarism_snippet":
                check_paragraph_result["plagiarism_snippet"]
            }]
        }

        app.logger.info("📤 回傳結果成功")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def rephrase_pdf_check(req, global_vector_db, global_embedding_model):

    app.logger.debug("📥 收到 PDF Rephrase 請求")
    app.logger.debug(f"📦 Request Content-Type: {req.content_type}")
    app.logger.debug(f"📦 Request files: {req.files}")

    if 'file' not in req.files:
        app.logger.warning("⚠️ 沒有收到 'file' 欄位")
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_file = req.files['file']
    original_filename = uploaded_file.filename
    app.logger.debug(f"📄 檔案名稱：{original_filename}")

    if not original_filename.lower().endswith('.pdf'):
        app.logger.warning("⚠️ 上傳的不是 PDF")
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        fixed_filename = "uploaded_pdf.pdf"
        os.makedirs(PDF_SAVE_DIR, exist_ok=True)
        saved_path = os.path.join(PDF_SAVE_DIR, fixed_filename)
        uploaded_file.save(saved_path)
        app.logger.info(f"✅ PDF 已儲存：{saved_path}")

        # 開始處理 PDF
        app.logger.info("🧠 開始使用 Gemini 處理 PDF 段落")
        api_key = os.getenv("GEMINI_APIKEY")
        app.logger.debug(f"🔐 使用 API 金鑰：{bool(api_key)}")
        processed_pdf = process_pdf(saved_path, api_key)
        app.logger.info(f"📄 處理完成頁數：{len(processed_pdf)}")

        tansfer_array_to_json(processed_pdf, PDF_SAVE_DIR, "data.json")

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error":
                        f"Failed to save or process PDF: {str(e)}"}), 500

    # 統計段落數量
    total_paragraph_count = sum(len(p) for p in processed_pdf if p)
    app.logger.info(f"📊 總段落數：{total_paragraph_count}")
    app.logger.info("🔍 開始檢測抄襲...")

    total_plagiarism_percentage = 0
    total_confidence_score = 0
    all_check_result = []
    original_text_and_plagiarism_snippet = []

    paragraph_count = 0
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                paragraph_count += 1
                app.logger.debug(f"🔎 檢測第 {paragraph_count} 段")
                check_result = cooperate_plagiarism_check(
                    user_text=paragraph,
                    vector_db=global_vector_db,
                    embedding_model=global_embedding_model)

                original_text_and_plagiarism_snippet.append({
                    "original_text":
                    paragraph,
                    "plagiarism_snippet":
                    check_result["plagiarism_snippet"]
                })

                all_check_result.append(check_result)
                total_plagiarism_percentage += check_result[
                    "plagiarism_percentage"]
                total_confidence_score += check_result["plagiarism_confidence"]

    avg_confidence_score = total_confidence_score / total_paragraph_count
    avg_plagiarism_percentage = total_plagiarism_percentage / total_paragraph_count

    result = {
        "plagiarism_percentage":
        round(avg_plagiarism_percentage, 2),
        "plagiarism_confidence":
        round(avg_confidence_score, 2),
        "original_text_and_plagiarism_snippet":
        original_text_and_plagiarism_snippet,
    }

    tansfer_array_to_json(all_check_result, PDF_SAVE_DIR,
                          "all_check_result.json")
    tansfer_array_to_json(result, PDF_SAVE_DIR, "result.json")

    app.logger.info("✅ 抄襲檢測完成")
    return jsonify(result)


def simulate_generate_pdf(req, global_vector_db, global_embedding_model):
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


# ✅ 若是直接執行此檔案，則啟動開發伺服器
if __name__ == "__main__":
    load_dotenv()
    app.run(host="0.0.0.0", port=8077, debug=True)
