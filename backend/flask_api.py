from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from model.check_paraphrase_content import *
from model.check_generate_content import *

from pdf_processor_paraphrase.pdf_cutter import *
from pdf_processor_generate.extract_for_test import *

from tool.json_to_array import *
from tool.array_to_json import *

from opencc import OpenCC

cc = OpenCC('s2t')  # s2t: simplified to traditional

import logging
# ✅ 初始化：讀取環境變數與定義全域常數
load_dotenv()

REQUIRED_ENV_VARS = [
    "OPENAI_APIKEY",
    "BACKEND_API_URL_TEXT_GENERATE",
    "BACKEND_API_URL_TEXT_PARAPHRASE",
    "BACKEND_API_URL_PDF_GENERATE",
    "BACKEND_API_URL_PDF_PARAPHRASE",
]

SOURCE_DIRS = [
    # 正式部署用
    "dataset/paraphrased_dataset/source/ccu",
    "dataset/paraphrased_dataset/source/nccu_2018",
    "dataset/paraphrased_dataset/source/nccu_2019",
    "dataset/paraphrased_dataset/source/ncu_2019",
    "dataset/paraphrased_dataset/source/ncu_2020",
    "dataset/paraphrased_dataset/source/nsyu_2019",
    "dataset/paraphrased_dataset/source/nycu",

    # # 測試用路徑
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ccu",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2018",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nsyu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nycu",
]

# 正式部署用
PDF_SAVE_DIR = "uploaded_pdfs"

# # 測試用路徑
# PDF_SAVE_DIR = "/mnt/Agents4Financial/PlagiarismDetector/backend/uploaded_pdfs"


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

    validate_env()

    print(
        "Building FAISS vector store with all files... (this may take a while)"
    )
    global_paraphrase_vector_db, global_paraphrase_embedding_model, _ = build_paraphrase_vector_db(
        SOURCE_DIRS)
    # global_paraphrase_vector_db, global_paraphrase_embedding_model = None, None

    # --- 路由定義區 ---
    @app.route('/')
    def home():
        return "Plagiarism Checker API is running!"

    @app.route(os.getenv("BACKEND_API_URL_TEXT_PARAPHRASE"), methods=['POST'])
    def upload_text_paraphrase():
        return paraphrase_text_check(request, global_paraphrase_vector_db,
                                     global_paraphrase_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_PARAPHRASE"), methods=['POST'])
    def upload_pdf_paraphrase():
        return paraphrase_pdf_check(request, global_paraphrase_vector_db,
                                    global_paraphrase_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_TEXT_GENERATE"), methods=['POST'])
    def upload_text_generate():
        return generate_text_check(request, global_paraphrase_vector_db,
                                   global_paraphrase_embedding_model)

    @app.route(os.getenv("BACKEND_API_URL_PDF_GENERATE"), methods=['POST'])
    def upload_pdf_generate():
        return generate_pdf_check(request, global_paraphrase_vector_db,
                                  global_paraphrase_embedding_model)

    return app


app = create_app()


def paraphrase_text_check(req, global_vector_db, global_embedding_model):
    app.logger.debug("📥 收到 Text Paraphrase 請求")

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
        tansfer_array_to_json(check_paragraph_result, PDF_SAVE_DIR,
                              "paraphrase_text_check_result.json")

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
            }],
            "verdict": {
                "result": check_paragraph_result["verdict"]["result"],
                "reason":
                cc.convert(check_paragraph_result["verdict"]["reason"])
            }
        }

        app.logger.info("📤 回傳結果成功")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def paraphrase_pdf_check(req, global_vector_db, global_embedding_model):

    app.logger.debug("📥 收到 PDF Paraphrase 請求")
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
        fixed_filename = "uploaded_paraphrased_pdf.pdf"
        os.makedirs(PDF_SAVE_DIR, exist_ok=True)
        saved_path = os.path.join(PDF_SAVE_DIR, fixed_filename)
        uploaded_file.save(saved_path)
        app.logger.info(f"✅ PDF 已儲存：{saved_path}")

        # 開始處理 PDF
        app.logger.info("🧠 開始使用 Gemini 處理 PDF 段落")
        api_key = os.getenv("GEMINI_APIKEY")
        app.logger.debug(f"🔐 使用 API 金鑰：{bool(api_key)}")

        processed_pdf = process_pdf(saved_path, api_key)
        tansfer_array_to_json(processed_pdf, PDF_SAVE_DIR, "data.json")

        # processed_pdf = tansfer_json_to_array(PDF_SAVE_DIR, "data.json")

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error":
                        f"Failed to save or process PDF: {str(e)}"}), 500

    # 統計段落數量
    total_paragraph_count = 0
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                total_paragraph_count += 1

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
    verdict = aggregate_document_verdict_simple(all_check_result)
    verdict["reason"] = cc.convert(verdict["reason"])

    result = {
        "plagiarism_percentage": round(avg_plagiarism_percentage, 2),
        "plagiarism_confidence": round(avg_confidence_score, 2),
        "original_text_and_plagiarism_snippet":
        original_text_and_plagiarism_snippet,
        "verdict": verdict,
    }

    tansfer_array_to_json(all_check_result, PDF_SAVE_DIR,
                          "all_check_result.json")
    tansfer_array_to_json(result, PDF_SAVE_DIR, "result.json")

    # result = tansfer_json_to_array(PDF_SAVE_DIR, "result.json")

    app.logger.info("✅ 抄襲檢測完成")
    return jsonify(result)


def generate_text_check(req, global_vector_db, global_embedding_model):
    app.logger.debug("📥 收到 Text Generate 請求")

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

        check_paragraph_result = detect_from_text(text)
        tansfer_array_to_json(check_paragraph_result, PDF_SAVE_DIR,
                              "result.json")

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
            }],
            # "verdict":
            # check_paragraph_result["verdict"],
        }

        app.logger.info("📤 回傳結果成功")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def generate_pdf_check(req, global_vector_db, global_embedding_model):

    app.logger.debug("📥 收到 PDF Generate 請求")
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
        fixed_filename = "uploaded_generate_pdf.pdf"
        os.makedirs(PDF_SAVE_DIR, exist_ok=True)
        saved_path = os.path.join(PDF_SAVE_DIR, fixed_filename)
        uploaded_file.save(saved_path)
        app.logger.info(f"✅ PDF 已儲存：{saved_path}")

        # 開始處理 PDF
        app.logger.info("🧠 開始處理 PDF 段落")
        processed_pdf = extract(saved_path)

        tansfer_array_to_json(processed_pdf, PDF_SAVE_DIR, "data.json")

    except Exception as e:
        app.logger.error(f"❌ 發生錯誤：{e}", exc_info=True)
        return jsonify({"error":
                        f"Failed to save or process PDF: {str(e)}"}), 500

    # 統計段落數量
    total_paragraph_count = len(processed_pdf)
    app.logger.info(f"📊 總段落數：{total_paragraph_count}")
    app.logger.info("🔍 開始檢測抄襲...")

    total_plagiarism_percentage = 0
    total_confidence_score = 0
    all_check_result = []
    original_text_and_plagiarism_snippet = []

    paragraph_count = 0
    for paragraph in processed_pdf:
        paragraph_count += 1
        app.logger.debug(f"🔎 檢測第 {paragraph_count} 段")
        check_result = detect_from_text(paragraph)

        original_text_and_plagiarism_snippet.append({
            "original_text":
            paragraph,
            "plagiarism_snippet":
            check_result["plagiarism_snippet"]
        })

        all_check_result.append(check_result)
        total_plagiarism_percentage += check_result["plagiarism_percentage"]
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


# ✅ 若是直接執行此檔案，則啟動開發伺服器
if __name__ == "__main__":

    load_dotenv()
    app.run(host="0.0.0.0",
            port=8077,
            debug=True,
            threaded=False,
            use_reloader=False)
