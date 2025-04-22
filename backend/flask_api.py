import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from dotenv import load_dotenv
from model.check_rephrase_content import *
from model.check_generate_content import *
from pdf_processor.pdf_cutter import *
from pdf_processor.twodarray_to_json import *
from pdf_processor.json_to_twodarray import *

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

# 正式部署用
# PDF_SAVE_DIR = "uploaded_pdfs"

# 測試用路徑
PDF_SAVE_DIR = "/home/undergrad/PlagiarismDetector/backend/uploaded_pdfs"


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


def rephrase_text_check(req, global_vector_db, global_embedding_model):
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


def rephrase_pdf_check(req, global_vector_db, global_embedding_model):
    # if 'file' not in req.files:
    #     return jsonify({"error": "No file part in the request"}), 400

    # uploaded_file = req.files['file']
    # original_filename = uploaded_file.filename

    # if not original_filename.lower().endswith('.pdf'):
    #     return jsonify({"error": "Only PDF files are supported"}), 400

    # try:
    #     # 指定固定檔名
    #     fixed_filename = "uploaded.pdf"

    #     os.makedirs(PDF_SAVE_DIR, exist_ok=True)

    #     # 儲存完整路徑
    #     saved_path = os.path.join(PDF_SAVE_DIR, fixed_filename)
    #     uploaded_file.save(saved_path)
    #     print("PDF uploaded and saved successfully")

    # except Exception as e:
    #     return jsonify({"error": f"Failed to save PDF: {str(e)}"}), 500

    # //////////////////////////////////////////////////////////////////////////

    # # 處理PDF文件
    # print(os.getenv("GEMINI_APIKEY"))
    # processed_pdf = process_pdf(saved_path, os.getenv("GEMINI_APIKEY"))

    # tansfer2darraytojson(processed_pdf, PDF_SAVE_DIR)

    processed_pdf = tansferjsonto2darray(PDF_SAVE_DIR)

    total_paragraph_count = 0
    total_paragraph_length = 0
    original_text = ""

    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                original_text += paragraph
                total_paragraph_count += 1
                total_paragraph_length += len(paragraph)
    print(f"總共有 {total_paragraph_count} 個段落")

    print("正在檢測各段落的抄襲情況")
    total_plagiarism_percentage = 0
    total_confidence_score = 0
    total_plagiarism_snippet = ""

    paragraph_count = 0

    temp_paragraph = ""
    temp_paragraph_length = 10

    all_result = []
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                original_text += paragraph
                paragraph_count += 1

                if (paragraph_count % temp_paragraph_length
                        == 0) or (paragraph_count == total_paragraph_count):
                    print(f"正在檢測第 {paragraph_count} 個段落")
                    check_paragraph_result = cooperate_plagiarism_check(
                        user_text=temp_paragraph,
                        vector_db=global_vector_db,
                        embedding_model=global_embedding_model)

                    total_paragraph_length += len(temp_paragraph)
                    print(temp_paragraph)
                    temp_paragraph = ""

                    all_result.append(check_paragraph_result)

                    total_plagiarism_percentage += check_paragraph_result[
                        "plagiarism_percentage"] * len(temp_paragraph)
                    total_confidence_score += check_paragraph_result[
                        "avg_confidence"] * len(temp_paragraph)
                    total_plagiarism_snippet += check_paragraph_result[
                        "plagiarism_snippet"]
                    total_plagiarism_snippet += "\n\n"
                else:
                    temp_paragraph += "\n\n"
                    temp_paragraph += paragraph

    # avg_confidence_score = total_confidence_score / total_paragraph_length
    # avg_plagiarism_percentage = total_plagiarism_percentage / total_paragraph_length

    # ✅ 刪除處理完的 PDF 檔案
    try:
        os.remove(saved_path)
        print(f"Deleted uploaded PDF: {saved_path}")
    except Exception as e:
        print(f"Warning: Failed to delete PDF: {str(e)}")

    return jsonify(all_result)

    # return jsonify({
    #     "plagiarism_percentage": avg_plagiarism_percentage,
    #     "plagiarism_snippet": total_plagiarism_snippet,
    #     "confidence_score": avg_confidence_score,
    #     "original_text": original_text,
    # })

    text = "這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應，這是測試的回應"
    plagiarism_percentage = round(random.uniform(0, 100), 2)
    confidence_score = round(random.uniform(0, 100), 2)
    plagiarism_snippet = text[:min(30, len(text))]

    return jsonify({
        "plagiarism_percentage": plagiarism_percentage,
        "plagiarism_snippet": plagiarism_snippet,
        "confidence_score": confidence_score,
    })


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


app = create_app()
# ✅ 若是直接執行此檔案，則啟動開發伺服器
if __name__ == "__main__":
    load_dotenv()
    app.run(host="0.0.0.0", port=8077, debug=True)
