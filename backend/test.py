import os
from dotenv import load_dotenv
from model.check_paraphrase_content import *
from model.check_generate_content import *

from pdf_processor_paraphrase.pdf_cutter import *
from pdf_processor_generate.extract_for_test import *

from tool.json_to_array import *
from tool.array_to_json import *
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
    # 測試用路徑
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ccu",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2018",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2019",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nsyu_2019",
    "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nycu",
]
# 測試用路徑
PDF_SAVE_DIR = "/mnt/Agents4Financial/PlagiarismDetector/backend/uploaded_pdfs"


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


if __name__ == "__main__":
    validate_env()

