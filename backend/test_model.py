import os
from dotenv import load_dotenv
from model.check_rephrase_content import *
from model.check_generate_content import *
from pdf_processor.pdf_cutter import *
from pdf_processor.json_to_array import *
from pdf_processor.array_to_json import *
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
    # "dataset/paraphrased_dataset/source/ccu",
    # "dataset/paraphrased_dataset/source/nycu",

    # 測試用路徑
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ccu",
    "/home/undergrad/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nycu"
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


def rephrase_pdf_check(global_vector_db, global_embedding_model):

    processed_pdf = tansfer_json_to_array(PDF_SAVE_DIR, "data.json")

    total_paragraph_count = 0
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                total_paragraph_count += 1

    print(f"總共有 {total_paragraph_count} 個段落")

    print("正在檢測各段落的抄襲情況")
    total_plagiarism_percentage = 0
    total_confidence_score = 0

    all_check_result = []

    original_text_and_plagiarism_snippet = []
    paragraph_count = 0
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                paragraph_count += 1
                print(f"正在檢測第 {paragraph_count} 個段落")

                check_paragraph_result = cooperate_plagiarism_check(
                    user_text=paragraph,
                    vector_db=global_vector_db,
                    embedding_model=global_embedding_model)

                temp_result = {
                    "original_text":
                    paragraph,
                    "plagiarism_snippet":
                    check_paragraph_result["plagiarism_snippet"],
                }

                all_check_result.append(check_paragraph_result)

                original_text_and_plagiarism_snippet.append(temp_result)

                total_plagiarism_percentage += check_paragraph_result[
                    "plagiarism_percentage"]
                total_confidence_score += check_paragraph_result[
                    "plagiarism_confidence"]

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


if __name__ == "__main__":
    validate_env()
    print(
        "Building FAISS vector store with all files... (this may take a while)"
    )
    global_vector_db, global_embedding_model, _ = build_vector_db(SOURCE_DIRS)

    # 處理PDF文件

    # processed_pdf = process_pdf(saved_path, os.getenv("GEMINI_APIKEY"))
    # tansfer_array_to_json(processed_pdf, PDF_SAVE_DIR, "data.json")

    rephrase_pdf_check(global_vector_db, global_embedding_model)
