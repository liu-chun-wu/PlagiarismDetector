import os
from dotenv import load_dotenv
from model.check_paraphrase_content import *
from model.check_generate_content import *
from pdf_processor_SogoChang.pdf_cutter import *
from pdf_processor_SogoChang.json_to_array import *
from pdf_processor_SogoChang.array_to_json import *
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


def list_all_pdfs(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(filename)  # 只保留檔名，不含路徑
    return pdf_files


def test_pdf_cutter(file_name):
    saved_path = os.path.join(PDF_SAVE_DIR, file_name)
    processed_pdf = process_pdf(saved_path, os.getenv("GEMINI_APIKEY"))

    total_paragraph_count = 0
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                total_paragraph_count += 1

    print(f"總共有 {total_paragraph_count} 個段落")

    print("正在合併段落")

    original_text_list = []

    paragraph_count = 0
    paragraph_combined_number = 4
    temp_paragraph = ""
    for page_idx, page_paragraphs in enumerate(processed_pdf):
        if page_paragraphs:
            for para_idx, paragraph in enumerate(page_paragraphs):
                paragraph_count += 1
                temp_paragraph += paragraph
                temp_paragraph += '\n'
                if paragraph_count % paragraph_combined_number == 0 or paragraph_count == total_paragraph_count:
                    original_text_list.append(temp_paragraph)
                    temp_paragraph = ""

    new_file_name = file_name.replace(".pdf", ".json")
    tansfer_array_to_json(original_text_list, PDF_SAVE_DIR, new_file_name)


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

    pdf_files = list_all_pdfs(PDF_SAVE_DIR)
    for pdf_name in pdf_files:
        test_pdf_cutter(pdf_name)

    # print(
    #     "Building FAISS vector store with all files... (this may take a while)"
    # )
    # global_vector_db, global_embedding_model, _ = build_vector_db(SOURCE_DIRS)

    # 處理PDF文件

    # rephrase_pdf_check(global_vector_db, global_embedding_model)
